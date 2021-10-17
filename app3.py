
from fastai.vision.data import *
from fastai.vision import *
import numpy as np
import torch
from fastai.imports import *
import pickle
import streamlit as st
import PIL
import torchvision.transforms as T

learn = load_learner ('') #Mention the Path to your weights file


#CATEGORY to reduce loading time
def category(tensor):
  if tensor == 0:
    return "Becker's Nevus"
  elif tensor == 1:
    return "Drug Reaction"
  elif tensor == 2:
    return "Elephantiasis"
  elif tensor == 3:
    return "Melanoma"
  elif tensor == 4:
    return "Milia"
  elif tensor == 5:
    return "Post inflamatory Hyperpigmentation"
  elif tensor == 6:
    return "Sebaceous Cyst"
  elif tensor == 7:
    return "Vitiligo"
  elif tensor == 8:
    return "Angioedema"
  elif tensor == 9:
    return "Aphthous Stomatitis"
  elif tensor == 10:
    return "Beau's Lines"
  elif tensor == 11:
    return "Callosity"
  elif tensor == 12:
    return "Cherry Angiomas"
  elif tensor == 13:
    return "Dermatitis"
  elif tensor == 14:
    return "Dermatosis Papulosa Nigra"
  elif tensor == 15:
    return "Erythema Nodosum"
  elif tensor == 16:
    return "Furunculosis"
  elif tensor == 17:
    return "Idiopathic Guttate Hypomelanosis"
  elif tensor == 18:
    return "Lichen Planus"
  elif tensor == 19:
    return "Lipoma"
  elif tensor == 20:
    return "Melanonychia"
  elif tensor == 21:
    return "Melasma"
  elif tensor == 22:
    return "Neurofibromatosis"
  elif tensor == 23:
    return "Nevus or Moles"
  elif tensor == 24:
    return "Nodulocystic Acne"
  elif tensor == 25:
    return "Onychomycosis"
  elif tensor == 26:
    return "Rosacea"
  elif tensor == 27:
    return "Seborrheic Keratosis"
  elif tensor == 28:
    return "Tinea Cruris or Tinea Pedis" 

#SYMPTOMS 
def Symptoms(category):
  if category.lower() == "melanoma":
    return "Common symptoms (सामान्य लक्षण): bigger mole diameter, darkening of the skin, mole color changes, or skin mole with irregular border. बड़ा तिल, त्वचा का काला पड़ना, तिल का रंग बदलना या अनियमित सीमा के साथ त्वचा का तिल होना"
  elif category.lower() == "angioedema":
    return "Common symptoms (सामान्य लक्षण): Painless swelling under the skin, triggered by an allergy to animal dander, pollen, drugs, venom, food or medication. त्वचा के नीचे दर्द रहित सूजन या पशु की रूसी, पराग, दवाओं, विष, भोजन या दवा से एलर्जी के कारण"
  elif category.lower() == "beau's lines":
    return "Common symptom (सामान्य लक्षण)s: deep groove lines on nails running from side to side.  नाखूनों पर गहरी रेखाएँ " 
  elif category.lower() == "becker's nevus":
    return "Common symptoms (सामान्य लक्षण): Absence of the pectoralis major muscle (pectoral) and Underdevelopment of the muscles of the shoulder girdle. पेक्टोरलिस प्रमुख पेशी (पेक्टोरल) की अनुपस्थिति" #
  elif category.lower() == "cherry angiomas":
    return "Common symptoms (सामान्य लक्षण): Bleeding occurs if scratched. खरोंचने पर खून निकलना"  
  elif category.lower() == "dermatitis":
    return "Common symptoms (सामान्य लक्षण): skin rash, itching. त्वचा के लाल चकत्ते और खुजली" 
  elif category.lower() == "elephantiasis":
    return "Common symptoms (सामान्य लक्षण): swelling or swollen lymph nodes. सूजे हुए लिम्फ गाँठ" 
  elif category.lower() == "callosity":
    return "Common symptoms (सामान्य लक्षण): A thick rough area on the skin. त्वचा पर एक मोटा कठोर क्षेत्र" 
  elif category.lower() == "erythema nodosum":
    return "Common symptoms (सामान्य लक्षण): flat, firm, hot, red, and painful lumps or inflammatory nodules of the leg. चपटा, दृढ़, गर्म, लाल और दर्दनाक गांठ या पैर की सूजन वाली गांठ"
  elif category.lower() == "melasma":
    return "Common symptoms (सामान्य लक्षण): darkening of the skin. त्वचा का काला पड़ना"
  elif category.lower() == "lipoma":
    return "Common symptom (सामान्य लक्षण): lump. गांठ"
  elif category.lower() == "melanonychia":
    return "Common symptoms (सामान्य लक्षण): over two thirds of the nail discolored. नाखून के दो तिहाई हिस्से का रंग उद जाना" 
  elif category.lower() == "neurofibromatosis":
    return "Common symptoms (सामान्य लक्षण): brown spot on skin, armpit freckles, or lumps and pain on the face. त्वचा पर भूरे धब्बे, बगल की झाइयाँ, या गांठ और चेहरे पर दर्द" 
  elif category.lower == "furunculosis":
    return "Common symptoms (सामान्य लक्षण): puss or tenderness. खरहा या कोमलता"
  elif category.lower() == "milia":
    return "Common symptoms (सामान्य लक्षण): rashes or small bump. चकत्ते या छोटी गांठ"
  elif category.lower() == "idiopathic guttate hypomelanosis":
    return "Common symptoms (सामान्य लक्षण): darkening of the skin. त्वचा का काला पड़ना"
  elif category.lower() == "nevus or moles":
    return "Common symptoms (सामान्य लक्षण): darkening of the skin. त्वचा का काला पड़ना" 
  elif category.lower() == "lichen planus":
    return "Common symptoms (सामान्य लक्षण): blister, darkening of the skin, loss of colour, peeling, rashes. छाला, त्वचा का काला पड़ना, रंग बिगाड़ना, छिल जाना, चकत्ते पड़ना" 
  elif category.lower() == "nodulocystic acne":
    return "Common symptoms (सामान्य लक्षण): inflamed and uninflamed nodules and frequently, scars. सूजन और असंक्रमित गाँठ और अक्सर, निशान"  
  elif category.lower() == "onychomycosis":
    return "Common symptoms (सामान्य लक्षण): discolouration, thickening, or brittle of the nails. नाखूनों का टूटना, मोटा होना या भंगुर होना" 
  elif category.lower() == "sebaceous cyst":
    return "Common symptoms (सामान्य लक्षण): skin cyst, lump with central blackhead, or mucousy drainage. त्वचा की पुटी, केंद्रीय ब्लैकहेड या श्लेष्मा जल निकासी के साथ गांठ"
  elif category.lower() == "post inflamatory hyperpigmentation":
    return "Common symptoms (सामान्य लक्षण): Doesn't have underlying causes. अंतर्निहित कारण नहीं है"
  elif category.lower() == "rosacea":
    return "Common symptoms (सामान्य लक्षण): dryness, oily skin, rashes that look like acne, or swollen blood vessels in the skin. शुष्कता, तैलीय त्वचा, चकत्ते जो मुँहासे की तरह दिखते हैं, या त्वचा में रक्त वाहिकाओं की सूजन"
  elif category.lower() == "tinea cruris or tinea pedis":
    return "Common symptoms (सामान्य लक्षण): itching, rashes, groin rash, dryness, fissures, peeling,,stinging or redness खुजली, ऊसन्धि की लाली, लालपन, चकत्ते, सूखापन, दरारें, छीलना या चुभने वाली सनसनी"
  elif category.lower() == "senile comedones":
    return "Common symptoms (सामान्य लक्षण): a small skin coloured papule found on the face of a middle-aged or older person. मध्यम आयु वर्ग या वृद्ध व्यक्ति के चेहरे पर पाया जाने वाला एक छोटा सा रंग का पप्यूल"
  elif category.lower() == "vitiligo":
    return "Common symptoms (सामान्य लक्षण): loss of skin colour or premature hair whitening. त्वचा का रंग खराब होना या समय से पहले बाल सफेद होना"
  elif category.lower() == "seborrheic  keratosis":
    return "Common symptoms (सामान्य लक्षण): itching, small bump on skin, or waxy elevated skin lesion. खुजली, त्वचा पर छोटी सी गांठ, या मोमी बढ़े हुए त्वचा का घाव" 
  elif category.lower() == "aphthous stomatitis":
    return " Common symptom (सामान्य लक्षण): a painful sore in the mouth that can make it hard to eat and talk. मुंह में एक दर्दनाक दाना जो खाने और बात करने में मुश्किल कर सकता है"
  elif category.lower() == "drug reaction":
    return " Common symptom (सामान्य लक्षण): Skin rash, Hives, Itching, Fever, Swelling, Shortness of breath.  त्वचा के लाल चकत्ते, खुजली, बुखार, सूजन, सांस फूलना"
  


#Main Web app code
def main():
  try:
    st.markdown("<h1 style='text-align: center; color: red;'>MediDerma (मेडीडर्मा)</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Skin Disease Prediction ( त्वचा रोग अनुसन्धान )</h2>", unsafe_allow_html=True)
    st.write("")
    st.sidebar.markdown("<h4 style='text-align: center; color: black;'><u>The Following Diseases can be Detected:</u></h4>", unsafe_allow_html=True)
    st.sidebar.write('')
    lis = ['Angioedema', 'Aphthous Stomatitis',"Beau's Lines","Becker's Nevus",'Callosity','Cherry Angiomas','Dermatitis',
            'Dermatosis Papulosa Nigra','Drug Reaction','Elephantiasis','Erythema Nodosum','Furunculosis','Idiopathic Guttate Hypomelanosis',
            'Lichen Planus','Lipoma','Melanoma','Melanonychia','Melasma','Milia','Neurofibromatosis','Nevus or Moles',
            'Nodulocystic Acne','Onychomycosis','Post inflamatory Hyperpigmentation','Rosacea','Sebaceous Cyst',
            'Seborrheic Keratosis','Tinea Cruris or Tinea Pedis','Vitiligo']
    for i in lis:
        st.sidebar.write(str(i))
    st.markdown("<h5 style='text-align: center; color: black;'>Check the sidebar for the list of detectable diseases along with the symptoms</h5>", unsafe_allow_html=True)
    st.write("")
    img_uploaded = st.file_uploader("Choose an Zoomed Image of Your Skin (अपनी त्वचा की एक ज़ूम की गई फ़ोटो चुनें ...)",type=["jpeg","jpg"])
    st.markdown("<h5 style='text-align: center; color: black;'>Attaching a faulty/unclear image may result in wrong prediction or no prediction at all. </h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>दोषपूर्ण / अस्पष्ट फ़ोटो लगाने पर गलत परिणाम मिल सकता है। </h5>", unsafe_allow_html=True)
    if img_uploaded is not None:
      img_pil = PIL.Image.open(img_uploaded)
      img_tensor = T.ToTensor()(img_pil)
      img_fastai = Image(img_tensor)
      st.image(img_pil, caption='Uploaded Image of the Skin (त्वचा की चुनी गई फ़ोटो)', use_column_width=True)
      st.write("")
      st.markdown("<h4 style='text-align: center; color: red;'>Classifying(फ़ोटो वर्गीकृत करी जा रही है)...</h4>", unsafe_allow_html=True)
      a,cat_tensor,c = learn.predict(img_fastai)
      o=category(int(cat_tensor))
      st.write("")
      st.markdown("<h4 style='text-align: center; color: black;'>Most Probably the Disease is (संभवतः इस बीमारी का नाम है) :</h4>", unsafe_allow_html=True)
      st.write("")
      st.write('')
      st.write(o)
      st.write('')
      j=Symptoms(o)
      st.write(j)
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.markdown("<h2 style='text-align: center; color: red;'>Attention / सावधान :</h2>", unsafe_allow_html=True)
      st.markdown("<h4 style='text-align: center; color: black;'>The prediction may not be 100% accurate. Please consult a doctor for further information before taking any medication.</h4>", unsafe_allow_html=True)
      st.write("")
      st.markdown("<h4 style='text-align: center; color: black;'>परिणाम शायद एकदम सही न हों । कोई भी दवा लेने से पहले अधिक जानकारी के लिए कृपया डॉक्टर से सलाह लें।</h4>", unsafe_allow_html=True)
      st.write("") 
      st.write("")
      st.write("")  
  except Exception:
    pass

if __name__ == '__main__':
  main()
