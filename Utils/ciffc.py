from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Starte Chrome (gebe ggf. den Pfad zum Chromedriver an)
driver = webdriver.Chrome()
driver.get('https://directory.ciffc.ca/')

# Warten bis Seite geladen (manuell anpassen oder explizit warten mit WebDriverWait)
time.sleep(10)  # Cloudflare Challenge abwarten

# Beispiel: Alle Kontaktkarten auslesen (Selector musst du ggf. anpassen)
contacts = driver.find_elements(By.CSS_SELECTOR, '.contact-card')

data = []
for contact in contacts:
    name = contact.find_element(By.CSS_SELECTOR, '.contact-name').text
    email = contact.find_element(By.CSS_SELECTOR, '.contact-email').text
    phone = contact.find_element(By.CSS_SELECTOR, '.contact-phone').text
    data.append({'Name': name, 'Email': email, 'Telefon': phone})

driver.quit()

# Speichern als Excel
df = pd.DataFrame(data)
df.to_excel('ciffc_contacts.xlsx', index=False)
