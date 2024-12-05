# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:34:11 2021

@author: admin
"""
import torch
import numpy as np 
import pandas as pd

from sklearn import metrics

from collections import defaultdict

from numpy import genfromtxt
from datetime import datetime


def load_triples(dataset):
    print("Loading facts from triples file")
    if dataset == 'yago':
        triples_path = "data/yago/train.txt"
    else:
        triples_path = "data/fb15k-237/train.txt"
        
    triples = np.loadtxt(triples_path, dtype=np.str)
    return triples
    
def load_dicts(dataset):

    print("Loading class_entity, entity_class mappings from types file")
    if dataset == 'yago':
        #type_file_path = "../benchmark/KGESemanticAnalysis-main/data/yago/yagoTransitiveType.tsv"
        # type_file_path = "../benchmark/KGESemanticAnalysis-main/data/yago/yagoSimpleTypes.tsv"
        type_file_path = "../benchmark/KGESemanticAnalysis-main/data/yago/SimpleTypeFactsWordnetLevel.tsv"
        
    else:
        type_file_path = "E:/course-phd/202009-project1/hlda/benchmark/KGESemanticAnalysis-main/data/freebase/freebaseTypes.tsv"

    class_entity_dict = defaultdict(set)
    entity_class_dict = defaultdict(set)

    #read yago Simple types
    if dataset == "yago":
        with open(type_file_path, 'r', encoding='utf-8') as yago_types:
            for line in yago_types:
                try:
                    entity, _, cl = line.split()
                    entity = entity.replace(">","").replace("<","")
                    cl = cl.replace(">","").replace("<","")
                    class_entity_dict[cl].add(entity)
                    entity_class_dict[entity].add(cl)
                except ValueError:
                    continue

    if dataset == "freebase":
        with open(type_file_path, "r") as fb_types:
            for line in fb_types:
                try:
                    entity, cl = line.split()
                    class_entity_dict[cl].add(entity)
                    entity_class_dict[entity].add(cl)
                except ValueError:
                    continue
    # with open(type_file_path, "r") as fb_types:
    #     for line in fb_types:
    #         try:
    #             entity, cl = line.split()
    #             class_entity_dict[cl].add(entity)
    #             entity_class_dict[entity].add(cl)
    #             #print(class_entity_dict)
    #         except ValueError:
    #             continue
    #         if (no ==10):
    #             print(class_entity_dict)
    #             print(entity_class_dict)
    #             #break


    print(len(class_entity_dict.keys()))
    class_entity_dict_df = pd.DataFrame([(k, list(v)) for k, v in class_entity_dict.items()],columns=['entity', 'type'])

    #class_entity_dict_df = pd.DataFrame.from_records(class_entity_dict, columns=['entity', 'type'])
    #print(class_entity_dict_df.shape[0])
    #print(class_entity_dict_df.head(10))
    #class_entity_dict_df.to_csv("class_entity_dict_df_test.csv", sep="\t", header=None, index = False)


    entity_class_dict = pd.DataFrame.from_records(entity_class_dict, columns=['entity', 'type'])
    #print(entity_class_dict.shape[0])
    #print(entity_class_dict.head(10))
    #entity_class_dict.to_csv("entity_class_dict.csv", sep="\t", header=None )
    
# TODO: choose the level 1 
# for i in range(len(class_entity_dict_df)):
#     if len(class_entity_dict_df.iloc[i].type) > 2000:
#         print(class_entity_dict_df.iloc[i].entity, len(class_entity_dict_df.iloc[i].type))

    return class_entity_dict, entity_class_dict


def get_types(input_classes, class_entity_dict):

    entity_types = []
    for type in class_entity_dict.keys():
        #print (type, len(class_entity_dict[type]))
        #check if the current type is present in the list of input classes
        if type in input_classes:
            print (len(class_entity_dict[type]), type)
            #now loop through all entities stored for this class, and add to df with the class as type
            for entity in class_entity_dict[type]:
                #print(entity)
                entity_types.append([entity, type])

    return entity_types


if __name__ == '__main__':
# def evaluation():
    #load the class names for experiments from files
    # datasets = {'yago','freebase'}
    datasets = {'freebase'}

    import collections
    experiments = collections.defaultdict(dict)

    experiments = {'yago':
        {
            'Level-1': ['wordnet_person_100007846', 'wordnet_organization_108008335', 'wordnet_body_of_water_109225146',
                        'wordnet_product_104007894'],
            'Level-2-Organizations': ['wordnet_musical_organization_108246613', 'wordnet_party_108256968',
                                      'wordnet_enterprise_108056231', 'wordnet_nongovernmental_organization_108009834'],
            'Level-2-Waterbodies': ['wordnet_stream_109448361', 'wordnet_lake_109328904', 'wordnet_ocean_109376198',
                                    'wordnet_bay_109215664', 'wordnet_sea_109426788'],
            'Level-2-Persons': ['wordnet_artist_109812338', 'wordnet_officeholder_110371450',
                                'wordnet_writer_110794014', 'wordnet_scientist_110560637',
                                'wordnet_politician_110450303'],
            # 'Level-3-Writers': ['wordnet_journalist_110224578', 'wordnet_poet_110444194', 'wordnet_novelist_110363573',
            #                     'wordnet_scriptwriter_110564905', 'wordnet_dramatist_110030277',
            #                     'wordnet_essayist_110064405', 'wordnet_biographer_109855433'],
            # 'Level-3-Scientists': ['wordnet_social_scientist_110619642', 'wordnet_biologist_109855630',
            #                        'wordnet_physicist_110428004', 'wordnet_mathematician_110301261',
            #                        'wordnet_chemist_109913824', 'wordnet_linguist_110264437',
            #                        'wordnet_psychologist_110488865', 'wordnet_geologist_110127689',
            #                        'wordnet_computer_scientist_109951070', 'wordnet_research_worker_110523076'],
            # 'level-3-Players': ['wordnet_football_player_110101634', 'wordnet_ballplayer_109835506',
            #                     'wordnet_soccer_player_110618342', 'wordnet_volleyball_player_110759047',
            #                     'wordnet_golfer_110136959'],
            # 'Level-3-Artists': ['wordnet_painter_110391653', 'wordnet_sculptor_110566072',
            #                     'wordnet_photographer_110426749', 'wordnet_illustrator_109812068',
            #                     'wordnet_printmaker_110475687']
        },

        'freebase':
            {
                # 'Level-1': ['wordnet_person_100007846', 'wordnet_organization_108008335',
                #             'wordnet_location_100027167',
                #             'wordnet_event_100029378'],
                'Level-1': ['wordnet_person_100007846', 'wordnet_organization_108008335'
                           ],
                'Level-2-Organizations': ['wordnet_institution_108053576'],
                'Level-2-Persons': ['wordnet_artist_109812338', 
                                'wordnet_leader_109623038'], # 5th try                
                # 'Level-2-Organizations': ['wordnet_institution_108053576'],
                # 'Level-2-Persons': ['wordnet_artist_109812338', 
                #                 'wordnet_writer_110794014'], # 4th try
                # 'Level-2-Persons': ['wordnet_artist_109812338', 'wordnet_entertainer_109616922',
                #                     'wordnet_communicator_109610660',
                #                     'wordnet_leader_109623038'], # 3rd try
                # 'Level-2-Organizations': ['wordnet_musical_organization_108246613', 'wordnet_party_108256968',
                #                           'wordnet_enterprise_108056231', 'wordnet_institution_108053576',
                #                           'wordnet_nongovernmental_organization_108009834'],
                # 'Level-2-Persons': ['wordnet_artist_109812338', 'wordnet_officeholder_110371450',
                #                 'wordnet_writer_110794014', 'wordnet_scientist_110560637',
                #                 'wordnet_politician_110450303'], # second try
                # 'Level-2-Persons': ['wordnet_artist_109812338', 'wordnet_entertainer_109616922',
                #                     'wordnet_communicator_109610660', 'wordnet_scientist_110560637',
                #                     'wordnet_leader_109623038'], # first try
                # 'Level-2-Locations':['wordnet_geographical_area_108574314','wordnet_region_108630039'],
                # 'Level-2-Event':['wordnet_show_106619065','wordnet_affair_107447261', 'wordnet_movie_106613686',
                #                  'wordnet_contest_107456188'],

            }
    }



    for dataset in datasets:

        try:
            class_entity_dict, entity_class_dict = load_dicts(dataset)
        except:
            print("Error reading classes")
            continue

        print("Loaded dataset: {}".format(dataset))
        
        level_2_all = pd.DataFrame(columns=['entities', 'classes'])
        for class_set in experiments[dataset].keys():
            input_classes = experiments[dataset][class_set]
            print("")
            print("Looking at classes from level:", class_set, input_classes)
            # print ("Class found:", input_classes)
            print("Finding entities for these classes...")

            print("Now finding entities for classes in ", input_classes)
            entity_types = get_types(input_classes, class_entity_dict)

            entity_types_df = pd.DataFrame(entity_types, columns=['entities', 'classes'])
            print(entity_types_df.head(5))
            print(entity_types_df.shape)

            # entity_types_df_unique = entity_types_df.drop_duplicates(subset=['entities'], keep = False)
            tmp = entity_types_df.loc[entity_types_df.duplicated(subset=['entities'], keep = False)]
            print('duplicate entities:\n', tmp)
            entity_types_df_unique = entity_types_df
            entity_types_df_unique.to_csv(class_set + '.csv')
            print('keep duplicates ',entity_types_df_unique.shape)
            # print('after drop duplicates ',entity_types_df_unique.shape)

            if 'Level-2' in class_set:
                level_2_all = level_2_all.append(entity_types_df_unique,ignore_index=True)
            else:
                level_1 = entity_types_df_unique

            type_freq = entity_types_df_unique.groupby(['classes']).size().sort_values(
                ascending=False).reset_index(name='count')
            print(type_freq)
            
    # return new_reference_data
        # start_time = datetime.now()
        # level_1_clip = pd.DataFrame()
        # entities = level_1.entities
        # for i in range(len(level_1)):
        #    tmpt = entities[i]
        #    if tmpt in level_2_all.entities.values:
        #        level_1_clip = level_1_clip.append(level_1.iloc[i])
        # end_time = datetime.now()
        # print('=======Duration: {}'.format(end_time - start_time))           
        # # time consuming & always stuck     
        # # level_1_clip = pd.DataFrame()      
        # # for i in range(len(level_2_all)):
        # #     tmpt = level_1[level_1.entities == level_2_all.entities[i]]
        # #     level_1_clip = level_1_clip.append(tmpt)
        # #      # cluster_entities(entity_embeddings_df_unique)
        # level_2_all_new = pd.DataFrame()
        # for i in range(len(level_2_all)):
        #     if level_2_all.entities[i] in level_1_clip.entities.values:
        #         level_2_all_new = level_2_all_new.append(level_2_all.iloc([i]))
        level_2_all_new = pd.DataFrame()
        entities = level_2_all.entities
        entities_1 = level_1.entities.values
        class1_list = []
        for i in range(len(level_2_all)):
            tmpt =  entities[i]
            if tmpt in  entities_1:
                level_2_all_new = level_2_all_new.append(level_2_all.iloc[i])
                index = list(entities_1).index(tmpt)
                class1_list.append(level_1['classes'][index])      
                
        level_2_all_new['level1classes'] = class1_list

        # level_1_clip.to_csv('level_1_clip.csv', index=False)
        datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        level_2_all_new.to_csv('data/fb15k-237/{}_reference_data_nhdp_{}.csv'.format(dataset,datetime), index=False)