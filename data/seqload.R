# Loading the package and ENSEMBL database
library(biomaRt)
mart<-useEnsembl(biomart="ensembl",dataset="hsapiens_gene_ensembl")

# Importing TF data
TF_IDs<-read.table("~/TFs_Ensembl_v_1.01.txt", quote="\"", comment.char="")
TF_IDs<-as.vector(unlist(TF_IDs))[-c(1,2)]
TF_UNIPROT<-getBM(attributes=c('ensembl_gene_id','uniprotswissprot'),filters ='ensembl_gene_id',values = TF_IDs,mart = mart,useCache = FALSE)
swiss_prot_ids<-unique(TF_UNIPROT$uniprotswissprot)[-3]

# Retrieving the sequences of TFs
all_seqs_TFs<-getSequence(id=swiss_prot_ids, type="uniprotswissprot", seqType="peptide", mart = mart)

# Importing random genes data
`%notin%` <- Negate(`%in%`)
DE<-read.csv("~/DE_results_TE_Hela_siEWS_vs_control.csv")
DE<-DE[DE$type=="protein_coding",]
random_ids<-sample(DE$X,2000,replace=FALSE)
random_ids<-random_ids[random_ids %notin% TF_IDs]
RANDOM_UNIPROT<-getBM(attributes=c('ensembl_gene_id','uniprotswissprot'),filters ='ensembl_gene_id',values = random_ids,mart = mart,useCache = FALSE)
swiss_prot_random_ids<-unique(RANDOM_UNIPROT$uniprotswissprot)[-3]
# Retrieving random genes sequences
all_seqs_random<-getSequence(id=swiss_prot_random_ids, type="uniprotswissprot", seqType="peptide", mart = mart)

# Remove genes with 2 or more sequences
remove_dup<-function(v)
{
  dup_names<-names(which(sort(table(v$uniprotswissprot),decreasing=TRUE)>1))
  duped<-v[v$uniprotswissprot %in% dup_names,]
  pos_dup<-0
  for (i in unique(duped$uniprotswissprot))
  {
    x1<-which(duped$uniprotswissprot==i)
    x2<-x1[2:length(x1)]
    pos_dup<-c(pos_dup,as.numeric(rownames(duped[x2,])))
  }
  pos_dup<-pos_dup[-1]
  return(v[-pos_dup,])  
}
unique_seq_TFs<-remove_dup(all_seqs_TFs)
unique_seq_random<-remove_dup(all_seqs_random)


# Saving the data in FASTA files
library(seqinr)

write.fasta(as.list(unique_seq_TFs$peptide),unique_seq_TFs$uniprotswissprot,"TF_seqs.fasta")
write.fasta(as.list(unique_seq_random$peptide),unique_seq_random$uniprotswissprot,"random_seqs.fasta")


# Retrivieving families data
library(readxl)
DatabaseExtract <- read_excel("C:/Users/loico/Downloads/DatabaseExtract_v_1.01 (1).xlsx")
DatabaseExtract<-as.matrix(DatabaseExtract)
DB2<-DatabaseExtract[which(DatabaseExtract[,5]=="Yes"),2:4]
colnames(DB2)<-c("ensembl_gene_id","HGNC","DBD")

# Merging with TF IDs

m_db<-merge(TF_UNIPROT,DB2,by="ensembl_gene_id")
m_db_uniprot<-m_db[,c(2,4)]
m_db_uniprot<-m_db_uniprot[which(m_db_uniprot[,1]!=""),]

# Saving data
write.table(m_db_uniprot,"families.txt",sep="\t",quote=FALSE,row.names = FALSE)



