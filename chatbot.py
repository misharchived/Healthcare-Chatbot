import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import csv
import warnings
from pyfiglet import Figlet

# speech recognition imports
import speech_recognition as sr
from gtts import gTTS
import os

sr.AudioFile.DEFAULT_FLAC_FILENAME = '/opt/homebrew/bin/flac'


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize gTTS engine
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")
    os.system("mpg321 temp.mp3")

def get_user_input(is_numeric=False):
    recognizer = sr.Recognizer()

    if not is_numeric:
        with sr.Microphone() as source:
            print("Say something!")
            audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio)
            print(f"User: {user_input}")

            return user_input.lower()

        except sr.UnknownValueError:
            speak("Sorry, I could not understand your audio.")
            return ""
        except sr.RequestError as e:
            speak(f"Error with the speech recognition service; {e}")
            return ""
    else:
        while True:
            try:
                user_input = input("Enter the numeric value: ")
                numeric_input = int(user_input)
                return numeric_input
            except ValueError:
                print("Please enter a valid numeric value.")

def get_user_input_confirmation():
    recognizer = sr.Recognizer()

    while True:
        try:
            # speak("Please respond with 'yes' or 'no' to confirm: ")
            with sr.Microphone() as source:
                print("Respond in positive or negative")
                audio = recognizer.listen(source)

            user_input = recognizer.recognize_google(audio)
            print(f"User: {user_input}")

            if user_input.lower() in ["positive", "negative"]:
                return user_input.lower()
            else:
                speak("Invalid response. Please respond with 'positive' or 'negative'.")

        except sr.UnknownValueError:
            speak("Sorry, I could not understand your audio.")
        except sr.RequestError as e:
            speak(f"Error with the speech recognition service; {e}")


def get_info():
    f = Figlet(font='slant')
    print(f.renderText('HealthCare ChatBot'))
    speak("HealthCare ChatBot. What's your name?")
    print("Your Name? \t\t\t\t", end="->")
    name = get_user_input()
    print(f"Hello, {name}")
    speak(f"Hello, {name}")


training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')

cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols] 
testy = testing['prognosis']
testy = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
print (scores)
print(scores.mean())

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum = sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('MainData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('MainData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MainData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    # print("-----------------------------------HealthCare ChatBot-----------------------------------")
    f = Figlet(font='slant')
    print(f.renderText('HealthCare ChatBot'))
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello, ",name)

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        # print("\nEnter the symptom you are experiencing  \t\t",end="->")
        # disease_input = input("")
        speak("Enter the symptom you are experiencing.")
        disease_input = get_user_input()
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        # if conf==1:
        #     print("searches related to input: ")
        #     for num,it in enumerate(cnf_dis):
        #         print(num,")",it)
        #     if num!=0:
        #         print(f"Select the one you meant (0 - {num}):  ", end="")
        #         conf_inp = int(input(""))
        #     else:
        #         conf_inp=0

        #     disease_input=cnf_dis[conf_inp]
        #     break
        # else:
        #     print("Enter valid symptom.")
        if conf == 1:
            speak("Did you mean any of the following?")
            for num, it in enumerate(cnf_dis):
                speak(f"{num}, {it}")
                print(num,")",it)

            if num != 0:
                speak(f"Select the one you meant (0 - {num}):")
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = get_user_input(is_numeric=True)
                # conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            speak("Enter a valid symptom.")

    while True:
        try:
            speak("Okay. From how many days ? (Enter number of days) ")
            # num_days=int(input("Okay. From how many days ? : "))
            num_days = get_user_input(is_numeric=True)
            # get_days()
            break
        except:
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            speak("Are you experiencing any ")
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                speak(f"{syms},? : ")
                # while True:
                #     # inp=input("")
                #     inp = get_user_input()
                #     if(inp=="yes" or inp=="no"):
                #         break
                #     else:
                #         print("provide proper answers i.e. (yes/no) : ",end="")
                #         speak("provide proper answers i.e. (yes/no) : ")
                # if(inp=="yes"):
                #     symptoms_exp.append(syms)

                while True:
                    speak("Provide proper answers using (positive/negative) : ")
                    inp = get_user_input_confirmation()

                    if inp == "positive" or inp == "negative":
                        break
                    else:
                        speak("Provide proper answers i.e. (positive/negative) : ")

                if inp.lower() == "positive":
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                # print("You may have ", present_disease[0])
                speak(f"You may have , {present_disease[0]}")
                # print(description_list[present_disease[0]])
                speak(description_list[present_disease[0]])


            else:
                # print("You may have ", present_disease[0], "or ", second_prediction[0])
                speak(f"You may have , {present_disease[0]}, or , {second_prediction[0]}")
                # print(description_list[present_disease[0]])
                speak(description_list[present_disease[0]])
                # print(description_list[second_prediction[0]])
                speak(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            speak("Take following measures : ")
            for i,j in enumerate(precution_list):
                print(i+1,")",j)
                speak(f"{i+1},),{j}")

    recurse(0, 1)
# def tree(tree, feature_names, symptom_input, num_days):
#     tree_ = tree.tree_
#     feature_name = [
#         feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]

#     chk_dis = ",".join(feature_names).split(",")
#     symptoms_present = []

#     while True:
#         speak("Enter the symptom you are experiencing.")
#         disease_input = get_user_input()
#         conf, cnf_dis = check_pattern(chk_dis, disease_input)
#         if conf == 1:
#             speak("Did you mean any of the following?")
#             for num, it in enumerate(cnf_dis):
#                 speak(f"{num}, {it}")
#             if num != 0:
#                 speak(f"Select the one you meant (0 - {num}):")
#                 conf_inp = int(get_user_input())
#             else:
#                 conf_inp = 0

#             disease_input = cnf_dis[conf_inp]
#             break
#         else:
#             speak("Enter a valid symptom.")

#     def recurse(node, depth):
#         indent = "  " * depth
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]

#             if name == disease_input:
#                 val = 1
#             else:
#                 val = 0
#             if val <= threshold:
#                 recurse(tree_.children_left[node], depth + 1)
#             else:
#                 symptoms_present.append(name)
#                 recurse(tree_.children_right[node], depth + 1)
#         else:
#             present_disease = print_disease(tree_.value[node])
#             red_cols = reduced_data.columns
#             symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
#             speak("Are you experiencing any ")
#             symptoms_exp = []
#             for syms in list(symptoms_given):
#                 inp = ""
#                 speak(f"{syms}? : ")
#                 while True:
#                     inp = get_user_input()
#                     if inp.lower() == "yes" or inp.lower() == "no":
#                         break
#                     else:
#                         speak("Provide proper answers i.e. (yes/no) : ")
#                 if inp.lower() == "yes":
#                     symptoms_exp.append(syms)

#             second_prediction = sec_predict(symptoms_exp)
#             calc_condition(symptoms_exp, num_days)
#             if present_disease[0] == second_prediction[0]:
#                 speak(f"You may have {present_disease[0]}")
#                 speak(description_list[present_disease[0]])
#             else:
#                 speak(f"You may have {present_disease[0]} or {second_prediction[0]}")
#                 speak(description_list[present_disease[0]])
#                 speak(description_list[second_prediction[0]])

#             precution_list = precautionDictionary[present_disease[0]]
#             speak("Take the following measures:")
#             for i, j in enumerate(precution_list):
#                 speak(f"{i + 1}. {j}")


#     recurse(0, 1)


def main():
    getSeverityDict()
    getDescription()
    getprecautionDict()
    get_info()
    tree(clf,cols)
    print("----------------------------------------------------------------------------------------")
    speak("Thank you for using HealthCare ChatBot. Have a great day!")


if __name__ == "__main__":
    main()


