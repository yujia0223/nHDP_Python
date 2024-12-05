
import time
import os
import rdflib
import json
import random
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON



#clusters = [
#    'Place', 'NaturalPlace', 'PopulatedPlace', 'BodyOfWater', 'Settlement', 'Island', 'Country', 'Lake', 'Sea', 'City', 'Town',
#    'Person', 'Artist', 'Athlete', 'Actor', 'MusicalArtist', 'Painter', 'SoccerPlayer', 'WinterSportPlayer', 'ClassicalMusicArtist', 'BeachVolleyballPlayer', 'IceHockeyPlayer'
#]

# clusters = [
#     'Lake', 'Mountain', 'City', 'Town', 'Island', 'Country',
#     'Actor', 'MusicalArtist', 'Painter', 'SoccerPlayer', 'AmericanFootballPlayer', 'IceHockeyPlayer', 'Swimmer'
# ]

clusters = [
    'Actor', 'SoccerPlayer'
]

# ########################## PART 1 ##########################

# for obj in clusters:
#     others = set(clusters)
#     others.remove(obj)
#     except_str = " ".join(["FILTER NOT EXISTS{?s ?r dbo:" + x + ".} " for x in others])


#     sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#     sparql.setQuery("""
#         PREFIX dbo: <http://dbpedia.org/ontology/>
#         PREFIX dbr: <http://dbpedia.org/resource/>
#         PREFIX dct: <http://purl.org/dc/terms/>
#         PREFIX dbp: <http://dbpedia.org/page/>
#         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#         SELECT ?s 
#         WHERE { 
#                 ?s rdf:type dbo:""" + obj + """
#                 """ + except_str + """
#             } ORDER BY RAND() LIMIT 1000""")
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()

#     counter = 0
#     with Path(obj + ".txt").open('w', encoding="utf-8") as output_file:

#         for triple in results['results']['bindings']:
#             output_file.write(str(triple['s']['value']) + "\n")
#             counter += 1
#     print(obj, counter)


########################## PART 2 ##########################

def query(candidates, line):
    sub = line.strip()

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX dbp: <http://dbpedia.org/page/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?r ?o 
        WHERE { 
                <""" + sub + """> ?r ?o
                FILTER NOT EXISTS{<""" + sub + """> rdf:type ?o.}
            } ORDER BY RAND() LIMIT 1000""")
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    for triple in results['results']['bindings']:

        if 'http://dbpedia.org/resource/' in str(triple['o']['value']):
            candidates.append(sub + '\t' + str(triple['r']['value']) + '\t' +  str(triple['o']['value']))
        #output_file.write(str(triple['r']['value']) + "\n")
    time.sleep(random.random())

    return candidates

candidates = []
for obj in clusters:
    with Path(obj + ".txt").open('r', encoding="utf-8") as input_file:
        lines = input_file.readlines()
        print(len(candidates))
        good = set()
        first_lines = lines[:150]
        while len(good) !=  len(first_lines):
            line = first_lines[0]
            try:
                query(candidates, line)
                good.add(line)
                first_lines.pop(0)
                print(obj, len(good))
            except:
                time.sleep(random.random() * 20)

random.shuffle(candidates)
print(len(candidates))
candidates = set(candidates)
print(len(candidates))

with Path("triples.txt").open('w', encoding="utf-8") as output_file:

    for triple in candidates:
        output_file.write(str(triple) + "\n")

# ########################## PART 3 ##########################

# levels = {
#     'Lake' : 'https://dbpedia.org/ontology/Place\thttps://dbpedia.org/ontology/NaturalPlace\thttps://dbpedia.org/ontology/BodyOfWater\thttps://dbpedia.org/ontology/Lake',
#     'Mountain' : 'https://dbpedia.org/ontology/Place\thttps://dbpedia.org/ontology/NaturalPlace\thttps://dbpedia.org/ontology/Mountain\thttps://dbpedia.org/ontology/Mountain', 
#     'City' : 'https://dbpedia.org/ontology/Place\thttps://dbpedia.org/ontology/PopulatedPlace\thttps://dbpedia.org/ontology/Settlement\thttps://dbpedia.org/ontology/City',
#     'Town' : 'https://dbpedia.org/ontology/Place\thttps://dbpedia.org/ontology/PopulatedPlace\thttps://dbpedia.org/ontology/Settlement\thttps://dbpedia.org/ontology/Town',
#     'Island' : 'https://dbpedia.org/ontology/Place\thttps://dbpedia.org/ontology/PopulatedPlace\thttps://dbpedia.org/ontology/Island\thttps://dbpedia.org/ontology/Island',
#     'Country' : 'https://dbpedia.org/ontology/Place\thttps://dbpedia.org/ontology/PopulatedPlace\thttps://dbpedia.org/ontology/Country\thttps://dbpedia.org/ontology/Country',
#     'Actor' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Artist\thttps://dbpedia.org/ontology/Actor\thttps://dbpedia.org/ontology/Actor',
#     'MusicalArtist' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Artist\thttps://dbpedia.org/ontology/MusicalArtist\thttps://dbpedia.org/ontology/MusicalArtist',
#     'Painter' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Artist\thttps://dbpedia.org/ontology/Painter\thttps://dbpedia.org/ontology/Painter',
#     'SoccerPlayer' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Athlete\thttps://dbpedia.org/ontology/SoccerPlayer\thttps://dbpedia.org/ontology/SoccerPlayer',
#     'AmericanFootballPlayer' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Athlete\thttps://dbpedia.org/ontology/GridironFootballPlayer\thttps://dbpedia.org/ontology/AmericanFootballPlayer',
#     'IceHockeyPlayer' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Athlete\thttps://dbpedia.org/ontology/WinterSportPlayer\thttps://dbpedia.org/ontology/IceHockeyPlayer',
#     'Swimmer' : 'https://dbpedia.org/ontology/Person\thttps://dbpedia.org/ontology/Athlete\thttps://dbpedia.org/ontology/Swimmer\thttps://dbpedia.org/ontology/Swimmer'
# }

# subjects = set()
# with Path("triples.txt").open('r', encoding="utf-8") as input_file:
#     lines = input_file.readlines()
#     for line in lines:
#         subjects.add(line.strip().split("\t")[0])

# added = set()
# with Path("classes.txt").open('w', encoding="utf-8") as output_file:
#     output_file.write('subject\tlevel1\tlevel2\tlevel3\tlevel4\n')
#     for class_ in levels:
#         with Path(class_ + ".txt").open('r', encoding="utf-8") as input_file:
#             lines = input_file.readlines()
#             for line in lines:
#                 if line.strip() in subjects:
#                     output_file.write(line.strip() + '\t' + levels[class_] + '\n')
#                     added.add(line.strip())

# print(added.difference(subjects))
# print(subjects.difference(added))

