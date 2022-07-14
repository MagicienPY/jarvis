import pyttsx3




def voval():
    engine = pyttsx3.init()
    
    text = input('')

    engine.say(text)
    engine.runAndWait()#instriction  fermeture

def dial():
    engine = pyttsx3.init()#initialisation
    engine.say("bonjour mon nom c'est")
    engine.say("CHADOMAGIQUE")
    engine.say("bonjour ")
    engine.say("comment ca vas  ")
    engine.say("entrez le texte a lire")

    engine.runAndWait()

#on appel nos fonctions
dial()
voval()