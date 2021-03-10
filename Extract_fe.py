import numpy as np
def read_sequence():
    label = []
    protein_seq_dict = {}
    protein_index = 0
    with open('E:\\PycharmProjects\\ACP-ALPM_master\\ACP500\\ACP_AAC+DPC\\sequence.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]
                proteinName = values[0]
                if label_temp == '1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1]
                protein_seq_dict[protein_index] = seq
                protein_index = protein_index + 1
    return label,protein_seq_dict

def DPC():
    dpc = {}
    dpc_index1 = 1
    with open('E:\\PycharmProjects\\ACP-ALPM_master\\ACP500\\ACP_AAC+DPC\\DPC.tsv', 'r') as t:
        for line in t:
            seq = line[:-1].split('\t')[1:]
            dpc[dpc_index1] = seq
            dpc_index1+=1
    for j in range(1,501):
        for i in range(len(dpc[j])):
            dpc[j][i]=float(dpc[j][i][:10])
    # print(dpc)
    dpc_fe = []
    for j in range(1, 501):
        dpc_fe.append(dpc[j])
    dpc_fe = np.array(dpc_fe)
    # print(dpc_fe.shape)
    return dpc_fe
DPC()

def fe():  #aac
    protein_seq_dict1 = {}
    protein_index1 = 1
    with open('E:\\PycharmProjects\\ACP-ALPM_master\\ACP500\\ACP_AAC+DPC\\AAC.tsv', 'r') as t:
        for line in t:
            seq = line[:-1].split('\t')[1:]
            protein_seq_dict1[protein_index1] = seq
            protein_index1= protein_index1 + 1
    # print(protein_seq_dict1)

    for j in range(1,501):
        for i in range(20):
            protein_seq_dict1[j][i]=float(protein_seq_dict1[j][i][:10])
            # print(protein_seq_dict1[j][i][:4])
    # print((protein_seq_dict1))
    fe = []
    for j in range(1, 501):
        fe.append(protein_seq_dict1[j])
    fe = np.array(fe)
    return fe
fe()

def PC_PseAAC():
    pcp=[]
    pc=[]
    with open('E:\\PycharmProjects\\ACP-ALPM_master\\ACP500\\ACP_PC-PseAAC.txt', 'r') as t:
        for line in t:
            if line[0] != '>':
                line = line[:-1].split('\t')
                pc.append(line)
    for i in range(500):
        temp=[]
        for j in range(len(pc[i])):
            temp.append(float(pc[i][j]))
        pcp.append(temp)
    pcp = np.array(pcp)
    # print(pcp.shape)
    return pcp
PC_PseAAC()

def BPF(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fea = []
    tem_vec =[]
    k = 10
    for i in range(k):
        if seq[i] =='A':
            tem_vec = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem_vec = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem_vec = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem_vec = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem_vec = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem_vec = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem_vec = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem_vec = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem_vec = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem_vec = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        fea = fea + tem_vec
    return fea

def get_bpf():
    label,protein_seq_dict=read_sequence()
    bpf = []
    for i in protein_seq_dict:
        bpf_feature = BPF(protein_seq_dict[i])
        bpf.append(bpf_feature)
        # protein_index = protein_index + 1
    bpf=np.array(bpf)
    # print(bpf.shape)
    return bpf
get_bpf()