#pip install pyttsx3
import pyttsx3


def inpu():
    engine = pyttsx3.init()
    
    text = input('')

    engine.say(text)
    engine.runAndWait()

def dial():
    engine = pyttsx3.init()
    engine.say("bonjour mon nom c'est")
    engine.say("CHADOMAGIQUE \a")
    engine.say("bonjour \a")
    
    engine.say("comment ca vas  ")
    engine.say("entrez le texte a lire")
   

    engine.runAndWait()

dial()
inpu()
