from nltk.chat.util import Chat, reflections
import pyttsx3
import time
import cv2
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
#--------------------------------------------------



def lancer():
    #cap = cv2.VideoCapture("[ OxTorrent.com ] Mr.Robot.S04E01.FRENCH.BDrip.XviD-EXTREME.avi")
    cap = cv2.VideoCapture(0)
    print("magapp-face en cours de lancement ........\n")
    print("quand vous voudrez sortir appuez sur S/s")
    a=1
    g=1

    if not (cap.isOpened()):
        print("impossible d'avoir la video")

    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    while True:
        _,image = cap.read()
       

        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      
        faces=face.detectMultiScale(image_gray,1.3,5)
        g=g+1
        print(g)
        for x,y, width, height in faces:
            good=cv2.rectangle(image,(x,y),(x+width,y+height),color=(255,1,4),thickness=7)
            print(good)
            a=a+1
            print(a)
            
        cv2.imshow('magicien RF', image)





        if cv2.waitKey(1)==ord('s'):
            print("voulez vous relancer Magapp-face\n")
            print(" l- pour relancer\n autre touche pour sortir\n")       
            a = input()
            if cv2.waitKey(1)==ord('l'):
                lancer()
            else :
                break
    
    cap.release()
    cv2.destroyAllWindows()

def perceptron():
    
    #ici on genère un dataset composé de 100 ligne et deux variables grace a la fonction make_blobs retrouve dans cklearn
    x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    print("la dimentions de x;", x.shape)
    print("la dimentions de y;", y.shape)

    plt.scatter(x[:,0], x[:, 1], c=y, cmap="summer")
    plt.show()


    #ici on cree des fonctions
    def initialisation(x):
        w = np.random.randn(x.shape[1], 1)
        b = np.random.randn(1)
        return (w, b)
    #ici je verifie si les shape correspondent a ce qui a ete entre
    #w, b =initialisation(x)
    #w.shape
    #b.shape


    def model(x, w, b):
        z = x.dot(w) + b
        #fonction d'activation      notons que shape affiche le dimentionement
        A = 1 / (1 + np.exp(-z))
        return A
    #A = model(x, w, b)
    #A.shape#normalement on doit avoir (100, 1) parce que on a 100 valeurs

    def log_loss(A, y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    #log_loss(A, y)

    def gradients(A, x, y):
        dw = 1 / len(y) * np.dot(x.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return dw, db

    #dw, db = gradients(A, x, y)
    #dw.shape #ici on affiche les dimention de dw

    def update(dw, db, w, b, learning_rate):
        w = w - learning_rate * dw
        b = b - learning_rate * db
        return (w, b)
    def predict(x, w, b):
        A = model(x, w, b)
        print(A)
        return A >= 0.5
    #fin des fonction de cette algorithe de decente du gradient

    def neurones(x, y, learning_rate=0.1, n_iter=100):
        #initialisation
        w, b = initialisation(x)
        loss = []

        for i in range(n_iter):
            A = model(x, w, b)
            loss.append(log_loss(A, y))
            dw, db = gradients(A, x, y)
            w, b = update(dw, db, w, b, learning_rate)

        y_pred = predict(loss)
        print(accuracy_score(y, y_pred))    
        plt.plot(loss)
        plt.show()

        return (w, b)

    w, b = neurones(x, y)




    new_plant = np.array([2, 1])

    x0 = np.linspace(-1, 4, 100)
    x1 = ( -w[0] * x0 - b ) / w[1]
    plt.scatter(x[:,0], x[:, 1], c=y, cmap="summer")
    plt.scatter(new_plant[0], new_plant[0],  c='r')
    plt.plot(x0, x1, c= 'aqua', lw = 5)
    plt.show()

    predict(new_plant, w, b)

    #jusqu'ici on a genere un neurone



def securite():
    name = 'moi'
    pas = 'geni'
    engine = pyttsx3.init()

    def nom():
        def engin():
            engine = pyttsx3.init()
            print("salut, vous ètes prié de vous identifier, pour continuer ")
            engine.say("salut, vous ètes prié de vous identifier, pour continuer ")
            engine.runAndWait()
            engin()
        print ("votre nom d'utilisateur")
        nom = input('')
        
    def passe():
        print ("\n votre mot de passe svp")
        passe = input('')
        nom()
        passe()
        if ( passe != pas):
            engine.say("bienvenue monsieur ")
            dial()
            voval()
        else:
            engine.say ("mot de passe ou nom érronné merci de refaire")
            securite()

   
    
    engine.runAndWait()
    
    
def voval(text):
    engine = pyttsx3.init()
    text = input('')
    engine.say(text)
    engine.runAndWait()#instriction  fermeture

def dial():
    b = 'oui'
    engine = pyttsx3.init()#initialisation
    engine.say("bonjour mon nom c'est")
    engine.say("CHADOMAGIQUE je suis pret a te suivre. que voulez vous faire? ")
    
    engine.say("j'ai une question! ")
    engine.say("soyaitez vous que je vous aides lire un text?")
    a = input('')

    if (a == b):
        engine = pyttsx3.init()
        engine.say("entrez document  le text a lire s'il vous plais ")
        print("\a ok \a")
        text1 = input('')
        engine.say(text1)
        engine.runAndWait()
        engine.runAndWait()
    else:
        print ("ok ")
        engine = pyttsx3.init()
        engine.say("ok que voulez vous donc tres chers , votre santé est ma prorité ")
        engine.runAndWait()
#from googletrans import Translator
        engine = pyttsx3.init()
        engine.say("bonjour je suis chadomagique")
        securite()
        print("voulez vous faire de la reconnaissance faciale")
        engine.say("voulez vous faire de la reconnaissace faciale?")
        gh = input()
        if( gh == b):
            securite()
            lancer()
        else:
            engine.say("erruer de lancement relancez s'il vous plais")
            chat()

        #vocal =engine( language =" fr -FR")
        #print (vocal)
        engine.runAndWait()


#on appel nos fonctions

#--------------------------------------------------
#--------------------------------------------------
def chat():
    ch="*+++++++++++++*"
    t="*c             *"
    a="*h             *"
    z="*a             *"
    e="*d             *"
    y="*o      ☻☻    *"
    u="*b             *"
    o="*o             *"
    r="*t             *"
    for i in range (15):
        
        print (ch[i],t[i],a[i],z[i],e[i],y[i],u[i],o[i],r[i],t[i],a[i],z[i],e[i],y[i],u[i],o[i],r[i],ch[i])
        
    for i in range (5):
        print ("\ \ \ \ \ ",ch[i],"---------",ch[i],"/ / / / /")
        #dialogue --------------------------------------------
    pairs =[
        ["mon nom c'est (.*)", ["bonjour %1 mon Nom c'est CHADO BOT "]], 
        ["bonjour|salut|coucou", ["salut", "hello", "coucou",'en qoi puis je vous aider ?']],
        ["veux rire|blague|racconte moi une blague|fait moi rire", ["quesqui est jaune et qui attend ?", "quand je suis noir je suis propres et quand je suis blanc je suis salle", "he ! toktok!",'en qoi puis je vous aider ?']],
        ["tableau|le tableau|qui es |c'est le tableau qui est propres|johnathan", ["woow genial tu a trouve", "bravo", "cool","c'est juste"]],
        ["je veux|j'ai besoin de (.*)",["ok je vais préparer  votre %1"]],
        ["trouve (.*)",["ok je vais trouver %1"]],
        ["cherche moi (.*)",["ok je vais trouver %1"]],
        ["cherche (.*)",["ok je vais trouver %1"]],
        ["qui a  (.*)",["je ne sais pas qui a %1"]],
        ["(.*)(ville|adresse)", ["ici on est au cameroun"]],
        ["(.*)(faim|manger)", ["de quoi avez vous besoin  "]],
        ["(.*)(plat|degené)", ["ok  on vous le raport"] ],
        ["j'aime (.*)",["ooh ok vous aimez  %1  doi je pensser a autre chose ?"]],
        ["(.*)(non|pas la pein)", ["daccord et que voulez vous ?"] ],
        ["(.*)(sommeil|dormire|j'ai sommeil|je veux dormir)", ["ok si vous voulez dormir reposez vous je suis la je vous attendrais"] ],
        ["(.*)(bien sur|oui)", ["ok  on vous l'aurais dans peut de temps merci de patienter..."] ],
        ["(.*)(humm|ok)", ["quesqui vous inqiette ?"] ],
        ["(.*)(gentil|tu est très gentil)", ["je sais c'est normal j'ai ete programmé pour ca"] ],
        ["(.*)(allez vous|fatiguée|epuisé)", ["domage comment palier a votre bien etre cher  ?"] ],
        ["(.*)(qui me cherche|cherche)", ["d'apres mes notes presonne ne vous a chercher si je ne me trompe pas"] ],
        ["(.*)(appelez moi|lance un appel a)", ["ete vous serieux ?  vous même vous faite quoi je suis votre esclave ?"] ],
        ["(.*)(etteind la télé|etteindre)", ["ok quoi d'a ?"] ],
        ["qui a (.*)",["je ne sais pas qui a %1"]],
        ["(.*)(comment tu vas|comment ca vas |tu vas)", ["je vais bien et toi "] ],
        ["(.*)(je vais male|male|je ne me sens pas bien|j'ai male|je souffre)", ["mince qu'y a t'il je vous es offencé ? "] ],
        ["(.*)(je vais bien|bien)", ["c'est bien si vous allez bien je suis fiere"] ],
        ["(.*)(je vais bien|bien merci)", ["ok je vois vous allez bien?"] ],
        ["(.*)(jus|coca|ananas|orange|pamplemous|pome)", ["ok on vous apporte votre boisson  savier vous que les sucrerie ne sont pas bien pour la santé ?"] ],
        ["(.*)(rien|ne te derange pas)", ["pour tant tu m'avais l'aire inquiette OK  je te fais confiance ..."] ],
        ["(.*)(merci|je vous en pris)", ["de rien c'est la moidre des choses"] ],
        ["(.*)(soife|boire|eau|boisson|fraicheur)", ["quel est la boisson que vous voulez ?"]],
        ["(.*)(hotel|dormir|someille|fatigue|fatigué)", ["voulez vous que je vous reserve une chanbre ?"]],
        ["(.*)(je ne sais pas|pas du tout)", ["ok je vais le faire a votre place","je vais le fait pour vous si vous le permettez "]],
        ["(.*)(au revoir|a plus)", ["d'accord on ce revois bientôt","ok a plus tard tres chers"]],
        ["(.*)(pain|poulet)", ["ok  on vous l'apport","ce serras tous ?"] ]

    ]
    chat=Chat(pairs, reflections)
    chat.converse()

    #sortie vocal
    #import pyttsx3
    #engine = pyttsx3.init()
    #engine.say("bonjour .my name  is, ngongo ,ngongo, didier. je suis prés a te servir .")

    #reconnaissance vocal
#dial()


def depart():
    def mr_mad(genre):
        if (genre == "h"):
            print ("bonjour monsieur ")
        elif (genre == "f"):
            print ("bonjour madame ")
        else:
            print ("movais choix")
    def salutation():
        engine = pyttsx3.init()
        engine.say("BONJOUR.")
        engine.runAndWait()

    def choix():
        engine = pyttsx3.init()
        #----------------------------------------------
        def logo():
            print ("             M.          .M          ")     
            print ("              MMMMMMMMMMM.           " )    
            print ("           .MMM\MMMMMMM/MMM.         ")     
            print ("          .MMM.7MMMMMMM.7MMM.        " )    
            print ("         .MMMMMMMMMMMMMMMMMMM        " )    
            print ("         MMMMMMM.......MMMMMMM       " )    
            print ("         MMMMMMMMMMMMMMMMMMMMM       " )    
            print ("    MMMM MMMMMMMMMMMMMMMMMMMMM MMMM  " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("   dMMMM.MMMMMMMMMMMMMMMMMMMMM.MMMMD " )  
            print ("    MMM8 MMMMMMMMMMMMMMMMMMMMM 8MMM  " )  
            print ("         MMMMMMMMMMMMMMMMMMMMM       " )  
            print ("         MMMMMMMMMMMMMMMMMMMMM       " )  
            print ("             MMMMM   MMMMM  magicien " )  
            print ("             MMMMM   MMMMM           " )  
            print ("             MMMMM   MMMMM           " )  
            print ("             MMMMM   MMMMM           " )  
            print ("             .MMM.   .MMM.           ") 
        logo()
            #----------------------------------------------
        print ("╔──────────────────────────────────────────────╗")
        print ("|          chadobot Alias My eyes                     |"         )
        print ("|      outil pour les aveugles et mal voyants         |"   )   
        print ("┖──────────────────────────────────────────────┙")
        print ("[1] assistance vocale                                     ")
        print ("[2] assistance ecrite                       ")
        print ("[3] reconnaissance faciale                               ")
        print ("[4] recherche faciale                              ")
        print ("[5] declancher l'intelligence artificiel CHADOBOT                                       ")
        print ("[q] QUITTER                                        ")
        chois = input ()

        if (chois == "1"):
            dial()
        elif (chois == "2"):
            chat()
        elif (chois == "3"):
            lancer()
        elif (chois == "4"):
            print("chois egale a 4");
        elif (chois == "5"):
            perceptron()
        else:
            quit
            #print ("sortir")

        

        engine.runAndWait()




    engine = pyttsx3.init()
    salutation()
    print("svp vous etes une femme ou un homme H/F ?\n")
    genre = input ()
    mr_mad(genre)
    engine.say("faite votre choix svp")
    choix()
    engine.runAndWait()