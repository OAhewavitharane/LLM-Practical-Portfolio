# How I used the LLM

I used **Microsoft Copilot** as my product-development partner, on 27 November 2026. Every prompt went into the same chat, so Copilot kept the earlier answers in mind and later prompts could just say "this product". I also used Copilot to generate the product photo, and **Claude** to make the app screens.

I first wrote longer prompts with every detail and rule in them, then cut them down to the short ones below. Each one still has the parts that matter: what I want, the context it needs, and a limit on the answer (6 reasons, 5 ideas, 3 risks) so the answers stay short and easy to compare.

## Prompt 1: the problem

```
Act as my product-development partner. Students often forget things like their ID card, charger, USB, books or water bottle when they rush between lectures and labs. Why does this happen? Give me 6 short reasons.
```

**What it gave:** rushing between classes; distractions such as conversations and notifications; poor organisation, with items spread across bags, desks and pockets; routines that change, because different classes need different things; memory overload; and assuming everything is packed, so you leave without checking.

**What I did with it:** kept it as the problem statement. Reasons 1 and 6 shaped the product the most, because both happen at the moment you leave a room. That's why PackSync warns you as you walk out, not only in the morning.

Screenshot: [01-prompt-problem.png](screenshots/01-prompt-problem.png)

## Prompt 2: ideas

```
Give me 5 product ideas to help students remember the things they need each day. Make some of them apps, some physical products, and some both. For each one, say how it works and one problem with it.
```

**What it gave:**

1. SmartPack, an app with a checklist for each lecture or lab. Problem: students may ignore the notifications.
2. SmartBag, a backpack with sensors and lights in each pocket. Problem: cost and charging.
3. LeaveCheck, a device by the door where you tap your ID or phone to see a checklist. Problem: it only works if you use it.
4. ClassMate, an app that reads your timetable and reminds you what to bring, like a charger and USB before a computer lab. Problem: the timetable has to be kept up to date.
5. PackSense, Bluetooth tags with an app that alerts you if a tagged item is missing when you leave. Problem: tags on books and bottles get expensive.

Copilot's pick was PackSense. It also showed pictures from the web next to the ideas, from softbank.jp, yankodesign.com, alibaba.com, dribbble.com and logitec.co.jp.

**What I did with it:** I didn't go with PackSense on its own. Tags only cover the items that have tags and don't know what I need on a particular day, while ClassMate knows what each class needs but can't tell if I left something behind. So I combined ideas 4 and 5. I didn't use the web pictures, because they are other companies' products and designs.

Screenshots: [02-prompt-ideas-01.png](screenshots/02-prompt-ideas-01.png), [02-prompt-ideas-02.png](screenshots/02-prompt-ideas-02.png), [02-prompt-ideas-03.png](screenshots/02-prompt-ideas-03.png)

## Prompt 3: combine, name and list features

```
I want to combine ideas 4 and 5. The app reads my timetable and tells me what I need each day, and small tags go only on the items I lose most, like my ID card, charger and USB. It should warn me if I walk away from a lab or lecture room and leave a tagged item behind. Give me 5 name ideas and 6 key features for this product.
```

**What it gave:** the names ClassGuard, PackSync, Don'tLeave, ClassTag and TakeMate, with PackSync as its favourite. The 6 features were timetable integration, a packing checklist for each class, Bluetooth item tags, leave-behind alerts, last-seen location and a daily readiness check. It also pointed out that a Bluetooth tag can only tell when an item is out of range, not which room it's in, so the app should use the timetable, the phone's location and the last Bluetooth connection together.

**What I did with it:** kept the name PackSync and all six features. The point about rooms went into the last-seen feature. Tagging only the items I lose most also gets round the cost problem Copilot gave for PackSense in prompt 2.

Screenshots: [03-prompt-combine-01.png](screenshots/03-prompt-combine-01.png), [03-prompt-combine-02.png](screenshots/03-prompt-combine-02.png)

## Prompt 4: target user

```
Who is the target user for this product? Describe one typical student who would use it in a short paragraph: their course, their daily routine and what they usually forget.
```

**What it gave:** a third-year Data Science undergraduate with lectures in the morning and computer labs in the afternoon, who moves between classrooms, the library and the cafeteria with a backpack holding their ID, laptop charger, USB drive, notebooks and water bottle, and sometimes leaves the ID card in a lab, the charger at home or the USB drive on a desk.

**What I did with it:** kept it as it is. It's close to my own timetable, so it was easy to check that it's realistic.

Screenshot: [04-prompt-target-user.png](screenshots/04-prompt-target-user.png)

## Prompt 5: risks

```
What are the 3 biggest problems or risks with this product, and how could we fix each one?
```

**What it gave:**

1. Bluetooth connection failures and false alerts. Fix: Bluetooth Low Energy tags, an adjustable alert distance and a short delay before an alert.
2. Students ignoring notifications. Fix: use the timetable to send only relevant reminders, and put the ID card and charger first.
3. Cost and battery life. Fix: a low-cost pack of 2–3 tags with long-lasting replaceable batteries.

It said Bluetooth reliability is the one to solve first, because wrong alerts would make students stop trusting the app.

**What I did with it:** all three fixes are in the design. I made the "short delay" 30 seconds and the starter pack 3 tags, one each for the ID card, charger and USB drive.

Screenshot: [05-prompt-risks.png](screenshots/05-prompt-risks.png)

## My own addition: smart notices

I wanted AI to be part of the product too, not only the tool I used to design it, so I added **smart notices**: an LLM in the app reads lecturers' notices and adds any items they mention to the right day's checklist, for example a calculator for a quiz. This is my idea, not Copilot's, so there is no screenshot for it. It is shown at the bottom of the first app screen.

## Prompt 6: product photo

```
Create a realistic product photo for a student product called PackSync. Show a smartphone on a desk with a packing checklist app on the screen, and three small white coin-sized Bluetooth tags: one clipped to a student ID card lanyard, one on a laptop charger cable and one on a USB drive keyring. Clean white background, soft lighting, modern product design style. No other brand names or logos.
```

**What it gave:** a photo of a phone showing a PackSync screen with today's classes (a Data Science lecture and a computer lab) and a packing checklist, next to three white round tags on an ID card holder, a charger cable and a USB drive keyring.

**What I did with it:** used it as the product photo, but it isn't perfect. The button says "All Set!" while the laptop, notebook and water bottle are still unticked. The charger has a two-pin European plug, not the three-pin plug used in Sri Lanka. It also added a plant and a laptop, even though I asked for a clean white background.

Screenshots: [06-prompt-product-image.png](screenshots/06-prompt-product-image.png), [06-product-image.png](screenshots/06-product-image.png)

## App screens with Claude

Generated images get small text and details wrong, so a photo alone can't show how the app works. I gave Claude the final feature list and asked for three phone screens: the morning checklist, a leave-behind alert and the tags page. It built them as a web page, which was saved as an image. I checked the screens against the features and the risk fixes. The 30-second delay, the 3-tag starter pack and the smart notice are all shown, and only the ID card, charger and USB drive have tags.

Screenshot: [07-app-screens-mockup.png](screenshots/07-app-screens-mockup.png)

## What I kept and what I changed

| From the LLM | Kept or changed |
|---|---|
| 6 reasons for forgetting | Kept as the problem statement |
| 5 ideas, with PackSense as its pick | Changed: combined ClassMate and PackSense |
| 5 names, with PackSync as its favourite | Kept PackSync |
| 6 key features | Kept all six and added smart notices |
| A tag can't tell which room it's in | Kept: the last-seen location uses the timetable, phone location and last connection |
| Target user | Kept |
| 3 risks and fixes | Kept all three; I chose a 30-second delay and 3 tags |
| Pictures from the web in its answers | Not used, because they are other people's products |
| Product photo | Kept, with its mistakes noted |
