## Metadata preparation for ADNI + A4 dataset
# the file was prepared from the data for all available scans for both datasets 
setwd('/Users/tagmarin/artagmaz/ad_dl/ADDL_pipeline/data/')

#### upload libraries ####
library(tidyr)
library(ggplot2)
library(gridExtra)
library(moments)
library(dplyr)

## data upload
#data = read.csv('idaSearch_9_27_2023.csv')
data = read.csv('idaSearch_2_06_2024.csv')
collection_info = read.csv("adni_a4_all_27_09_2023_2_06_2024.csv")
visit_dict = read.csv('VISITS.csv')

### replace visit code based on the dictionary

data$VISCODE = NA
data$Image.Data.ID = NA
for(i in 1:nrow(data)){
  id = paste0('[I,D]',data$Image.ID[i],'$')
  rows = grep(id, collection_info$Image.Data.ID)
  data$Image.Data.ID[i] =collection_info$Image.Data.ID[rows]
  data$VISCODE[i] =  collection_info$Visit[rows]
}

write.csv(data,'data_with_viscode.csv')
data = read.csv('data_with_viscode.csv')

## Description simplification
data$modality_subtype = NA
# ADNI PET
adni_pet = data[data$Project == 'ADNI' & data$Modality == 'PET',]

adni_pet$modality_subtype[grepl('AV45',adni_pet$Imaging.Protocol)] = 'AV45'
adni_pet$modality_subtype[grepl('AV45',adni_pet$Description)] = 'AV45'
adni_pet$modality_subtype[grepl('PIB',adni_pet$Description)] = 'PIB'
adni_pet$modality_subtype[grepl('PIB',adni_pet$Imaging.Protocol)] = 'PIB'
adni_pet$modality_subtype[grepl('FBB',adni_pet$Description)] = 'FBB'
adni_pet$modality_subtype[grepl('FBB',adni_pet$Imaging.Protocol)] = 'FBB'

adni_pet$modality_subtype[grepl('AV1451',adni_pet$Description)] = 'AV1451'
adni_pet$modality_subtype[grepl('AV1451',adni_pet$Imaging.Protocol)] = 'AV1451'
adni_pet$modality_subtype[grepl('Tau',adni_pet$Description)] = 'AV1451'
adni_pet$modality_subtype[grepl('TAU',adni_pet$Description)] = 'AV1451'

adni_pet$modality_subtype[grepl('FDG',adni_pet$Imaging.Protocol)] = 'FDG'
adni_pet$modality_subtype[grepl('FDG',adni_pet$Description)] = 'FDG'

for(i in 1:nrow(adni_pet)){
  if(is.na(adni_pet$modality_subtype[i]) == TRUE){
    adni_pet$modality_subtype[i] = adni_pet$modality_subtype[adni_pet$Subject.ID == adni_pet$Subject.ID[i] &
                                              adni_pet$Study.Date == adni_pet$Study.Date[i] &
                                                adni_pet$Type == 'Original']
  }
  
  if (adni_pet$Type[i] == 'Processed') {
    adni_pet$Imaging.Protocol[i] = sapply(
      strsplit(
        sapply(
          strsplit(adni_pet$Imaging.Protocol[adni_pet$Subject.ID == adni_pet$Subject.ID[i] &
                                               adni_pet$Study.Date == adni_pet$Study.Date[i] &
                                               adni_pet$Type == 'Original'],
                   split=';', fixed=TRUE),
          '[',2),
        '='),
      '[',2)
  }
}

#ADNI MRI
adni_mri = data[data$Project == 'ADNI' & data$Modality == 'MRI',]
adni_mri$modality_subtype[adni_mri$Type == 'Original'] = adni_mri$Description[adni_mri$Type == 'Original']
adni_mri$modality_subtype[adni_mri$Type == 'Processed'] = sapply( strsplit(adni_mri$Description[adni_mri$Type == 'Processed'],split=' <- ', fixed=TRUE), "[", 2 )

for(i in 1:nrow(adni_mri)){
 if (adni_mri$Type[i] == 'Processed') {
   adni_mri$Imaging.Protocol[i] = sapply(
      strsplit(
        sapply(
          strsplit(adni_mri$Imaging.Protocol[adni_mri$Subject.ID == adni_mri$Subject.ID[i] &
                                               adni_mri$Study.Date == adni_mri$Study.Date[i] &
                                               adni_mri$Type == 'Original'],
                   split=';', fixed=TRUE),
          '[',5),
        '='),
      '[',2)[1]
  }
}

#A4 PET
a4_pet = data[data$Project == 'A4' & data$Modality == 'PET',]
a4_pet$modality_subtype = sapply( strsplit(a4_pet$Description,split=' <- ', fixed=TRUE), "[", 1 )
a4_pet$modality_subtype[a4_pet$modality_subtype == 'Florbetapir'] = 'AV45'
a4_pet$modality_subtype[a4_pet$modality_subtype == 'Flortaucipir'] = 'AV1451'

# A4 MRI
a4_mri = data[data$Project == 'A4' & data$Modality == 'MRI',]
a4_mri$modality_subtype[grepl('T1',a4_mri$Description)] = 'T1'
a4_mri$modality_subtype[grepl('T2',a4_mri$Description)] = 'T2'
a4_mri$modality_subtype[grepl('FLAIR',a4_mri$Description)] = 'FLAIR'
a4_mri$modality_subtype[grepl('DWI',a4_mri$Description)] = 'DWI'
a4_mri$modality_subtype[grepl('fMRI_rest',a4_mri$Description)] = 'fMRI_rest'

data_proc  = rbind(adni_mri,adni_pet, a4_pet, a4_mri)
data_proc = data_proc[order(data_proc$Subject.ID),]
data_proc = data_proc[,-1]


write.csv(data_proc,'ADNI_A4_processed_02_06_2024.csv')

meta= read.csv('ADNI_A4_processed_02_06_2024.csv')

## data exploration
# collection_info = read.csv('adni_a4_all_27_09_2023_10_03_2023.csv')
# 
# collection_info$Dataset = NA
# 
# collection_info[collection_info$Visit == 'SCV2' | collection_info$Visit == 'SCV4',]$Dataset = 'A4'
# collection_info[collection_info$Visit != 'SCV2' & collection_info$Visit != 'SCV4',]$Dataset = 'ADNI'
# 
# 
# adni = separate(collection_info[collection_info$Dataset == 'ADNI' & collection_info$Modality == "PET",], 
#                 Description,into = c('Description','Description2'),sep = ' ' )
# adni = adni[adni$Description %in% c('AV45','FBB','AV1451','PIB'),]
# 
# adni = adni[,-9]
# 
# collection_info = rbind(collection_info[collection_info$Dataset == 'A4',],
#                         collection_info[collection_info$Dataset == 'ADNI' & collection_info$Modality == "MRI",]
#                         ,adni)
# 
# collection_sel = collection_info[,c("Dataset","Subject","Group","Sex",
#                                     "Age","Visit","Modality","Description")]
# 
# collection_sel[collection_sel$Modality == 'MRI',]$Description = 'MRI'
# collection_sel = unique(collection_sel)
# collection_sel = collection_sel[order(collection_sel$Subject),]
# barplot(table(collection_sel$Description))
# 
# abeta = collection_sel[collection_sel$Description %in% c('AV45','FBB','Florbetapir','PIB'),]
