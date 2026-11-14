# Demo video script (about 2 minutes: screen recording with voice-over)

**0:00–0:15 · The problem** *(on screen: the HeartPost title and disclaimer)*
"Almost half of US counties have no cardiologist. In rural areas, a nurse practitioner at a small clinic has to decide which patients with chest pain to send on an 87-mile round trip. HeartPost helps with that decision. It's a prototype, not medical advice."

**0:15–1:15 · The demo** *(on screen: the app in a phone-sized window)*
1. Click **"Man, 46 · Long Beach VA"** to get the red zone. "This is a real patient the model never saw. The bars show why it chose red: no chest pain, which is the strongest sign in this data, then sex and cholesterol. And here's the real angiography result: he had heart disease. The model got it right."
2. Click **"Woman, 55 · Cleveland"** to get the yellow zone. "She had heart disease too. With one threshold for everyone, the model would have called her green. Our separate threshold for women caught her."
3. Click **"Woman, 47 · Hungary"** to get the green zone with a red box. "And here the model fails. It put her in the green zone, but she had heart disease. Young women with non-typical pain are its weak spot, and the app says so on screen."
4. Press **Copy referral note**, then switch **EN → ES**. "The result becomes a referral note. It works offline, in English or Spanish, and no patient data leaves the device."

**1:15–1:45 · How it works** *(on screen: scroll to "Every patient the model never saw")*
"Here are all 678 test patients the model never saw. Sick patients are above the line, healthy ones below. We trained on Cleveland only. In three new hospitals, the textbook 13-test model missed 93 sick patients and ours missed 43, and all three baselines did worse. Watch what happens when I drag the threshold: catching more sick patients costs more false alarms. Now I switch off the separate threshold for women… and 10 sick women turn red, missed. That's the gap we found and fixed."

**1:45–2:00 · Limitation and next step** *(on screen: the "Where it fails" table)*
"It still misses young patients with atypical pain, and its data is from the 1980s. Next, we want to test it on present-day rural data and with real clinicians."
