#!/usr/bin/env python3

import argparse
import os
import sys
import unittest

from collections import defaultdict
from collections import namedtuple

import vcf
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC

DEBUG = False
#DEBUG = True
class Test(unittest.TestCase):
    def setUp(self):
        with open('test.vcf','r') as handle:
            self.reader=list(vcf.Reader(handle))
        self.sample='102517-21'
        self.vcf_file='test.vcf'
        self.ref_file='NC_011083.fasta'
        with open(self.ref_file,'r') as ref:
            self.reference=list(SeqIO.parse(ref,"fasta"))[0]

    def tearDown(self):
        pass

    # Stop after the first testcase fails
    # http://sookocheff.com/post/python/halting-unittest-execution-at-first-error/
    def run(self,result=None):
        if not result.errors:
            super(Test, self).run(result)

    def test_fetch_reference(self):
        ref=fetch_reference(self.vcf_file)
        self.assertEqual(ref,'NC_011083.fasta')

    def test_fetch_reference_fail(self):
        self.assertRaises(FileNotFoundError,fetch_reference,self.ref_file)

    def test_homref(self):
        result=[homref(record) for record in self.reader]
        expected=[True,True,True,True,False,True,True,False,True,False,True,False,True,False,False]
        self.assertEqual(result,expected)

    def test_symbolic(self):
        record=self.reader[0]
        self.assertTrue(symbolic(record.ALT[0]))
        self.assertFalse(symbolic(record.REF))

    def test_is_snip(self):
        result=[is_snip(record,self.sample) for record in self.reader]
        expected=[False,False,False,False,True,False,False,False,False,False,False,False,False,True,False]
        self.assertEqual(result,expected)

    def test_is_insertion(self):
        result=[is_insertion(record,self.sample) for record in self.reader]
        expected=[False,False,False,False,False,False,False,False,False,True,False,True,False,False,True]
        self.assertEqual(result,expected)

    def test_is_deletion(self):
        result=[is_deletion(record,self.sample) for record in self.reader]
        expected=[False,False,False,False,False,False,False,True,False,False,False,False,False,False,False]
        self.assertEqual(result,expected)
        
    def test_is_indel(self):
        result=[is_indel(record,self.sample) for record in self.reader]
        expected=[False,False,False,False,False,False,False,True,False,True,False,True,False,False,True]
        self.assertEqual(result,expected)

    def test_genotype_homref(self):
        # First we get all homref records
        homref_records=[record for record in self.reader if homref(record)]
        
        result=[ genotype_homref(record,self.sample,self.reference) for record in homref_records]
        expected=['G','AA','AA','AATGA','A','TGGTG','G','A','G']
        self.assertEqual(result,expected)
        #result=[genotype_homref(record,self.sample,self.reference) for record in self.reader if homref(record)]
        #print(result)
    
    def test_genotype_snip(self):
        # Get all snips
        snip_records=[record for record in self.reader if is_snip(record,self.sample)]
        expected=['A','G']
        result=[genotype_snip(record,self.sample) for record in snip_records]
        self.assertEqual(result,expected)

    def test_genotype_deletion(self):
        # Get all deletions
        del_records=[record for record in self.reader if is_deletion(record,self.sample)]
        expected=['A---------']
        result=[genotype_deletion(record,self.sample) for record in del_records]
        self.assertEqual(result,expected)

    def genotype_insertion(self):
        # Get all insertions
        ins_records=[record for record in self.reader if is_insertion(record,self.sample)]
        expected=['G']
        result=[genotype_insertion[record,self.sample] for record in ins_records]
        self.assertEqual(result,expected)

    def test_failed_none(self):
        DP=0
        MIN_DP=0
        f=False
        result=[failed(record,self.sample,DP,MIN_DP) for record in self.reader]
        expected=[f,f,f,f,f,f,f,f,f,f,f,f,f,f,f]
        self.assertEqual(result,expected)

    def test_failed_all(self):
        DP=9999
        MIN_DP=99999
        t=True
        result=[failed(record,self.sample,DP,MIN_DP) for record in self.reader]
        expected=[t,t,t,t,t,t,t,t,t,t,t,t,t,t,t]
        self.assertEqual(result,expected)

    def test_failed_all_homref(self):
        DP=20
        MIN_DP=99999
        t=True
        f=False
        result=[failed(record,self.sample,DP,MIN_DP) for record in self.reader]
        expected=[t,t,t,t,f,t,t,f,t,f,t,t,t,t,t]
        self.assertEqual(result,expected)

    def test_failed_default(self):
        DP=20
        MIN_DP=20
        t=True
        f=False
        result=[failed(record,self.sample,DP,MIN_DP) for record in self.reader]
        expected=[f,t,t,f,f,f,f,f,f,f,f,t,t,t,t]
        self.assertEqual(result,expected)

    def test_failed_dp_5(self):
        ## The lowest depth is 5, so nothing fails here
        DP=5
        MIN_DP=5
        t=True
        f=False
        result=[failed(record,self.sample,DP,MIN_DP) for record in self.reader]
        expected=[f,f,f,f,f,f,f,f,f,f,f,f,f,f,f]
        self.assertEqual(result,expected)

    def test_failed_dp_6(self):
        ## The lowest depth is 5, so one record fails here
        DP=6
        MIN_DP=6
        t=True
        f=False
        result=[failed(record,self.sample,DP,MIN_DP) for record in self.reader]
        expected=[f,t,f,f,f,f,f,f,f,f,f,f,f,f,f]
        self.assertEqual(result,expected)

    def test_genotype_record_default(self):
        DP=20
        MIN_DP=20
        result=[genotype_record(self.reference,record,self.sample,DP,MIN_DP).gt for record in self.reader]
        expected=['G','NN','NN','AATGA','A','A','TGGTG','A---------','G','G','A','N','N','N','NNNN']
        self.assertEqual(result,expected)

    def test_insert_full_list(self):
        target=[None]*5
        seq=[1]*5
        pos=0
        insert(seq,pos,target)
        self.assertEqual(insert(seq,pos,target),seq)

    def test_insert_full_list(self):
        target=[None]*5
        seq='11111'
        pos=0
        insert(seq,pos,target)
        self.assertEqual(target,list(seq))

    def test_insert_one(self):
        target=[None]*5
        seq='1'
        pos=4
        insert(seq,pos,target)
        expected=[None,None,None,None,'1']
        self.assertEqual(target,expected)

    def test_insert_indexerror(self):
        target=[None]*5
        seq='1'
        pos=5
        self.assertRaises(IndexError,insert,seq,pos,target)


# TODO: check if we got a g.vcf file instead of a vcf file
def fetch_reference(vcf_file):
    """ Read a vcf file and return the reference file used """
    with open(vcf_file,'r') as fin:
        line=fin.readline()
        # TODO: must all headers be at the top according to the vcf spec?
        while line.startswith('#'):
            if line.startswith('##reference=file'):
                return line.strip()[19:]
            line=fin.readline()
        raise FileNotFoundError("Unable to extract reference from vcf file")
            
def homref(record):
    """ Return wether or not a g.vcf record is a homref block """
    return 'END' in record.INFO

def symbolic(genotype):
    """ Return wether a genotype is symbolic per vcf standard """
    # vcf standard defines a symbolic allele as an angle-bracketed ID String 
    # TODO: use genotype.sequence again and add exception to handle _SV
    #gt=genotype.sequence
    gt=str(genotype)
    return gt.startswith('<') and gt.endswith('>')

def is_snip(record,sample):
    """ Is this g.vcf record a snip """

    # We consider a record a snip if the *actual call* is a snip
    # So something like this
    # REF=AGGG, ALT=[A, AGGGG, ATCG,<NON_REF>]
    # can be a snip if the actual call is ATCG

    # If its a homref block, its not a snip
    if homref(record):
        return False


    # As snip is defined as a ref and alt of equal length
    # So maybe 'mutation' would be a better term
    sample_call=record.genotype(sample).gt_bases
    ref_call=record.REF

    if len(sample_call) == len(ref_call):
        return True
    else:
        return False

def is_insertion(record,sample):

    # We consider a record an insertion if the *actual call* is an 
    # insertion.
    # So something like this
    # REF=AGGG, ALT=[A, AGGGG, ATCG,<NON_REF>]
    # can be an insertion if the actual call is AGGGG

    # a record is an insertion if the sample call is longer then the ref
    sample_call=record.genotype(sample).gt_bases
    ref_call=record.REF

    if len(sample_call) > len(ref_call):
        return True
    else:
        return False

def is_deletion(record,sample):

    # We consider a record a deletion if the *actual call* is a 
    # deletion.
    # So something like this
    # REF=AGGG, ALT=[A, AGGGG, ATCG,<NON_REF>]
    # can be an deletion if the actual call is A

    # a record is a deletion if the sample call is shorter then the ref
    sample_call=record.genotype(sample).gt_bases
    ref_call=record.REF
    if len(sample_call) < len(ref_call):
        return True
    else:
        return False

def is_indel(record,sample):
    """ Return wether a g.vcf record is an indel """

    return is_insertion(record,sample) or is_deletion(record,sample)

def genotype_homref(record,sample,reference):
    begin=record.POS-1 #vcf uses 1-based index
    end=record.INFO['END']
    seq=str(reference.seq[begin:end])
    return seq

def genotype_snip(record,sample):
    call=record.genotype(sample)
    return call.gt_bases

def genotype_deletion(record,sample):
    # For a deletion, a vcf file includes the position before the event.
    # So we return the 'ALT', padded with N for each nucleotide that was deleted
    call=record.genotype(sample)
    seq=call.gt_bases
    del_size=len(record.REF)-len(seq)
    seq+='-'*del_size
    return seq

def genotype_insertion(record,sample):
    # If we leave out insertions, the output fasta will be the same length
    # (and aligned to) the reference
    #print("WARN: discarding insertion on position {}".format(record.POS),file=sys.stderr)
    return record.REF

def failed(record,sample,DP,MIN_DP):
    if homref(record): #only homrefs have MIN_DP
        record_depth=record.genotype(sample)['MIN_DP']
        threshold_depth=MIN_DP
    else:
        try:
            record_depth=record.genotype(sample)['DP']
            threshold_depth=DP
        except AttributeError as e:
            print("Warning, no DP for sample '{}' in {}".format(sample,record),file=sys.stderr)
            return True
    if record_depth < threshold_depth:
        return True
    else:
        return False

def genotype_record(reference,record,sample,DP,MIN_DP):
    """ Take a g.vcf record and reference, and return the associated genotype """

    if record.CHROM != reference.id: # The record and reference dont match
        raise RuntimeError("Fasta reference {} and vcf reference dont match".format(reference.id,record.CHROM))

    # Next there are four option, which we treat differently
    # its a homref block
    # This is the call made for sample
    sample_call=record.genotype(sample).gt_bases
    ref_call=record.REF
    
    if homref(record):
        seq=genotype_homref(record,sample,reference)
    elif is_snip(record,sample):
        seq=genotype_snip(record,sample)
    elif is_deletion(record,sample):
        seq=genotype_deletion(record,sample)
    elif is_insertion(record,sample):
        # If we leave out insertions, the output fasta will be the same length
        # (and aligned to) the reference
        seq=genotype_insertion(record,sample)
    else:
        raise SyntaxError("Record {} is neither snip, indel nor homref, giving up".format(record))

    Genotype = namedtuple('Genotype', 'pos gt')
    # If the record didn't pass our quality standards
    if failed(record,sample,DP,MIN_DP):
        return Genotype(record.POS-1,'N'*len(seq))
    else:
        return Genotype(record.POS-1,seq)

def get_samples(vcf_file):
    with open(vcf_file,'r') as vcf_handle:
        vcf_reader=vcf.Reader(vcf_handle)
        return vcf_reader.samples
        
def insert(seq,pos,target):
    """ Unpack seq and insert it into target at index pos """
    seq=list(seq) #seperate all items
    seqlen=len(seq)
    for i in range(seqlen):
        target[i+pos]=seq[i]

def yield_genotypes(vcf_file,reference_file,DP=20,MIN_DP=None):
    """ Generator, yields Bio.SeqRecords based on the g.vcf file

        Pseudo code:
        for entry in reference_fasta:
            for sample in vcf_file:
                for record in vcf_file
                    seq+=genotype(record)
                yield SeqRecord(seq,headers)
        
    """
    if not MIN_DP:
        MIN_DP=DP

    samples=get_samples(vcf_file)
    with open(reference_file,'r') as ref:#, open(vcf_file,'r') as vcf_handle:
        for reference_record in SeqIO.parse(ref,"fasta"):
            for sample in samples: # parse the whole vcf file once for each sample
                ref_len=len(reference_record.seq)
                sequences=[None]*ref_len # genotype list for sample
                with open(vcf_file,'r') as vcf_handle:
                    vcf_reader=vcf.Reader(vcf_handle)
                    for record in vcf_reader:
                        pos,seq=genotype_record(reference_record,record,sample,DP,MIN_DP)
                        insert(seq,pos,sequences)
                        #sequences.append(seq)

                if DEBUG:
                    if None in sequences:
                        sequences=['X' if seq == None else seq for seq in sequences]
                else:
                    assert None not in sequences

                sequence=''.join(sequences)
                if reference_record.id.endswith('|'): # To preserve key|value in fasta header
                    ref_id=reference_record.id[:-1]
                header=' '.join((sample,ref_id,'DP',str(DP),'MIN_DP',str(MIN_DP)))
                # Create SeqRecord object
                record=SeqRecord(Seq(sequence,IUPAC.unambiguous_dna),header)
                                #id='sample|'+sample+'|'+reference_record.id
                                #description="mapped against {}".format(reference_record.id)
                yield record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a genotype vcf file to fasta')
    parser.add_argument('vcf_file', 
                        metavar='vcf_file',
                        type=str,
                        nargs='+',
                        help = ("Specify one or more g.vcf files. "
                        "Note, all files should be mapped against the same reference")
    )
    parser.add_argument('--reference', 
                        metavar='fasta_file', 
                        type=str,
                        nargs='*',
                        help = "Specify one or more reference files"
    )
    parser.add_argument('--output',
                        type=str,
                        nargs=1,
                        help = "Specify an output fasta file"
    )
    parser.add_argument('--DP', default=20, 
                        type=int, 
                        help="Records below this depth are returned as no data (N)"
    )
    parser.add_argument('--MIN_DP',default=None,
                        type=int,
                        help="Same as DP, but specific for homref blocks. DP will be used when not specified"
    )

    # unittest
    if len(sys.argv) == 1:
        parser.print_usage()
        print("\n### UNITTEST ###\n")
        unittest.main()
        #print(dir(parser))
        #print(parser.help)
        exit()
    
    args=parser.parse_args()

    # Make sure the output file doesnt exist
    if args.output: 
        args.output=args.output[0] #nargs returns a list of 1 item
        if not DEBUG and os.path.exists(args.output):
            raise FileExistsError("File {} exists".format(output))

    for filename in args.vcf_file:
        print("Parsing {}".format(filename),file=sys.stderr)
        # First, we try to find the reference for that vcf file
        try:
            ref=fetch_reference(filename)
        except FileNotFoundError:
            print("Extracting the reference file from the vcf failed, please specify it manually",file=sys.stderr)
            exit(-1)

        # If an output file wasnt specified
        # We base the output filename on the input vcf file
        # And we make sure it doesn't exist already
        if not args.output: # If the output file was specified
            output=os.path.splitext(filename)[0]
            output+='.fasta'
            if not DEBUG and os.path.exists(output):
                raise FileExistsError("File {} exists".format(output))
        else:
            output=args.output

        # write fasta
        with open(output,'a') as out:
            for seq_record in yield_genotypes(filename,ref,args.DP,args.MIN_DP):
                SeqIO.write(seq_record, out, "fasta")

        if DEBUG:
            print("WARNING: DEBUG IS ENABLED",file=sys.stderr)
