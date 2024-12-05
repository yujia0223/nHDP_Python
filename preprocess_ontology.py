from owlready2 import *
from pathlib import Path
#zz = onto_path.append("C:/Users/Marcin/Desktop/work/project2/new/IIMB_LARGE/000/onto.owl")

onto = get_ontology("data/IIMB_LARGE/000/onto.owl").load()

print(list(onto.classes()))
for i in onto.instances():
    print(i)

classes_dict = {}
for type_class in onto.classes():
    classes_dict[type_class] = list(onto.search(subclass_of = type_class))


subsumption_axioms = []
for class1 in classes_dict:
    if len(classes_dict[class1]) > -1:
        #print(class1)
        possible_parents = []
        for class2 in classes_dict:
            if class1 in classes_dict[class2]:
                if class1 != class2:
                    possible_parents.append(class2)
        if len(possible_parents) == 0:
            #print('no parent', class1)
            subsumption_axioms.append(('DUMMY.root', class1))

        elif len(possible_parents) == 1:
            subsumption_axioms.append((possible_parents[0], class1))
        else:
            print(class1, possible_parents)
            subsumption_axioms.append((possible_parents[-1], class1))

with Path('IIMB_subsumption_axioms').open('w', encoding="utf-8") as output_file:
    for axiom in subsumption_axioms:
        print(str(axiom[0]).split(".")[-1] + " " + str(axiom[-1]).split(".")[-1])
        output_file.write(str(axiom[0]).split(".")[-1] + " " + str(axiom[-1]).split(".")[-1]+'\n')
   



quit()
type_dict = {}

for type_class in onto.classes():
    type_class_string = str(type_class).split(".")[-1]
    print(type_class_string)
    entities = [str(entity).split(".")[-1] for entity in list(onto.search(type = type_class))]
    print(entities)

    for entity in entities:
        if entity not in type_dict:
            type_dict[entity] = ['root', type_class_string]
        else:
            type_dict[entity].append(type_class_string)

for entity in type_dict:
    print(entity, type_dict[entity])
    '''
    p = Path("IIMB/")
    p.mkdir(parents=True, exist_ok=True)
    fn = str(entity)
    filepath = p / fn

    with filepath.open('w', encoding="utf-8") as output_file:
        output_file.write(' '.join(type_dict[entity]))
    '''
