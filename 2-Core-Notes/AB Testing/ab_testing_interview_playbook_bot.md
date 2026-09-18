# A/B Testing Interview Playbook — Chatbot Credit Line Increase

**How to use this doc**
- **Part 1** is what you say out loud (about 5–7 minutes). Rehearse this.
- **Part 2** explains the bot flow and the key numbers.
- **Part 3** goes step by step, with follow-up questions and simple answers. Use it only when the interviewer asks for more.

---

## Part 1 — The Spoken Answer (rehearse this)

**Opening line:**
> "I'd walk through this in 7 steps: check if the idea is worth it, pick the metrics, figure out how many users we need, set up who gets which version, run the test and check the data is clean, look at the results, and decide whether to launch."

**Step 1 — Is it worth it?**
> "Today the bot answers credit line increase requests with a plain text link, which is easy to miss. We want to test an interactive card with a clear button, a short 'what happens next' note, and a line saying it won't affect their credit score. If it lifts submissions by 5%, that's worth about $250K a year."

**Step 2 — Metrics**
> "The main metric is the percent of users who submit a request within 7 days. I'd also track clicks to understand why it moved. And I'd watch 'do no harm' metrics (guardrail metrics), mainly how often users get transferred to a live agent, since agents are expensive."

**Step 3 — How many users, how long**
> "Today 25% of users submit. To detect a 5% lift, with the standard 5% false alarm risk (alpha) and 80% chance of catching a real win (power), I need about 19,000 users per group. At our traffic, that's about 2 weeks, which also covers both weekdays and weekends. Then I wait 7 more days so everyone has time to submit."

**Step 4 — Who gets which version**
> "Users who already have a pending request are left out first. Everyone else is randomly split 50/50 by user ID, so each person always sees the same version. We log the same events in both groups: saw the message, clicked, submitted, got transferred to an agent."

**Step 5 — Is the data clean?**
> "Before launch, I check the two groups look similar. During the test, I check the groups are close to 50/50. If not, something is broken. And I don't stop early just because results look good on Day 3."

**Step 6 — Results**
> "Submissions went from 25% to 26.8%, about a 7% lift. The p-value is very small, so it's a real effect, not luck. That's worth about $360K a year. Agent transfers didn't go up."

**Step 7 — Decision**
> "I'd launch, but gradually: 10%, then 50%, then 100%, watching agent transfers at each step. I'd keep a small group on the old version for a few months to confirm the lift lasts."

**5 numbers to remember:** 25% today · 5% target lift · ~19,000 users per group · 2 weeks · ~$360K/year

---

## Part 2 — The Bot Flow

### Today's Flow (Control)
```
User: "I want to increase my credit line"
            │
   Bot understands the request
            │
   Does the user already have a pending request?
      │                         │
     YES                        NO
      │                         │
 "You already have a      "You can submit a credit line increase
  pending request."        request by clicking here. We'll let you
  (left out of the test)   know if additional information is needed."
                                   │
                          Plain text link  ← CONTROL (old version)
                                   │
              User clicks → fills out form → submits (or not)

 At any point: if the bot can't help → transfer to a live agent
```

### New Version (Treatment)
Same flow, but instead of the plain text link, the bot shows an **interactive card** with:
- A clear button: **"Request a credit line increase"**
- A short note on what happens next (e.g., how long a decision takes)
- A reassurance line: **"This uses a soft credit check, so it won't affect your credit score."** (Final wording approved by Compliance.)

**Why it might work:** A text link is easy to miss in a chat. Users also worry that asking for more credit will hurt their score, so they hesitate or ask an agent. The card makes the next step obvious and answers that worry upfront.

### Key Numbers

| What | Number |
|---|---|
| Users asking for a credit line increase (no pending request) | ~2,800/day, ~500,000/year |
| Submit today (Control) | 25% |
| Submissions that get approved | ~40% |
| Value of one approved increase | ~$100/year |
| Transferred to an agent today | 10% |
| Cost of one agent chat | ~$8 |
| Smallest lift worth launching | 5% (25% → 26.25%) |
| Users needed | ~19,000 per group (~38,000 total) |
| How long | 2 weeks + 7 days to wait for submissions |

---

## Part 3 — Step-by-Step Detail and Follow-Up Questions

### Step 1 — Is It Worth It?

**Goal:** Make sure the change is worth building and testing.

**The math (say it slowly):**
- 500,000 users a year, 25% submit today.
- A 5% lift means 25% → 26.25%, or about **6,250 more submissions**.
- 40% get approved → **2,500 more approvals**.
- Each is worth ~$100/year → **~$250K/year**.

**Say it simply:** "I estimate what a realistic lift is worth. The smallest lift that's clearly worth it becomes my target."

#### Follow-Up Questions

**Q: "Why test at all? A button is obviously better than a link."**
A: Not always. The card could fail to show on some screens, or get more clicks from people who aren't ready and then ask for an agent. Testing shows the real overall effect.

**Q: "When would you NOT run an A/B test?"**
A: When there aren't enough users to see a meaningful change, when it's a required legal wording change, or when it's a clear bug fix. Then I'd compare before vs. after and keep an eye on the numbers.

---

### Step 2 — Metrics

**Goal:** Win on the thing that matters without hurting something else.

**1. Main metric (decides the test)**
**Submission rate** = % of users who **submit a request within 7 days** of seeing the bot message.
- Count submissions from **anywhere**: bot, app, or website. If the card convinces someone and they submit in the app, that still counts.

**2. Supporting metrics (explain why it moved)**
- Click rate on the link or button
- Of those who clicked, how many finished the form
- How many users leave the chat right after the message

**3. "Do no harm" metrics (guardrail metrics): must not get worse**
- **Agent transfer rate:** % of users sent to a live agent within 30 minutes. The most important one, since agents are expensive.
- **Repeat questions:** Users asking about credit increases again within 7 days (a sign of confusion).
- **Approval rate:** If lots of extra requests come in but most get declined, that's frustrating for customers.
- **Bot errors:** The card failing to show, slow responses.

**Say it simply:** "One metric decides the test. A few explain it. A few make sure we didn't cause more agent transfers or confusion."

#### Follow-Up Questions

**Q: "Why not use clicks as the main metric?"**
A: A button will almost surely get more clicks than a link. That doesn't mean more people finish. Clicks explain the result; submissions decide it.

**Q: "Why not use approvals or revenue?"**
A: The bot doesn't decide approvals; the credit team does. And revenue takes months to show up. So I track approvals as a "do no harm" (guardrail) metric and revenue in the long-term group.

**Q: "Why is agent transfer the key guardrail metric?"**
A: The bot's job is to help without a human. If the card confuses people and they ask for an agent, we lose savings, and longer wait times hurt every customer.

---

### Step 3 — How Many Users and How Long

**Goal:** Get enough users to trust the result.

**What goes into the calculation:**
- **Today's rate:** 25%
- **Smallest lift worth finding:** 5% (25% → 26.25%)
- **False alarm risk (alpha):** 5%. We accept a 5% chance of saying it works when it doesn't.
- **Chance of catching a real win (power):** 80%. If the lift is real, we'll spot it 80% of the time.

**Result:** ~19,000 users per group, ~38,000 total.

**How long:**
- ~2,800 users/day → about **2 weeks**.
- 2 full weeks covers weekdays and weekends (bot traffic differs by day).
- Then **wait 7 more days** so everyone has time to submit.

**We check in both directions:** Does the card help *or* hurt? We don't assume it can only help.

**Say it simply:** "The number of users depends on today's rate, the smallest lift I care about, and how sure I want to be."

#### Follow-Up Questions

**Q: "What if you want to catch a smaller lift, like 2.5%?"**
A: You'd need about **4 times the users** (~76,000 per group), so about 8 weeks. Smaller effects need a lot more data.

**Q: "What if bot traffic is too low?"**
A: Either aim for a bigger lift, run longer, or look at clicks as an early signal, while being clear that clicks aren't the final answer.

**Q: "Why wait 7 days for submissions?"**
A: Some users click, leave, and come back later to finish. 7 days catches most of them. I'd check past data to confirm.

---

### Step 4 — Who Gets Which Version

**Goal:** Put the right users in, split them fairly, and track the right things.

**Before the split:**
1. User is logged in (we need their ID and account info).
2. Bot recognizes the credit line increase request.
3. **Already has a pending request?** Show "You already have a pending request" and leave them out of the test.

This check happens **before** the split, the same way for everyone, so both groups stay comparable.

**The split:**
- Randomly assign by **user ID**, 50/50.
- Each person always sees the same version, even if they come back later.

**What we track (same in both groups):**
1. Saw the bot message
2. Clicked the link or button
3. Submitted the request (from any channel)
4. Got transferred to an agent (with the time)

**Count each user once**, starting from the first time they saw the message.

**Say it simply:** "Leave out users with a pending request, split everyone else 50/50 by user ID, and track the same four events in both groups."

#### Follow-Up Questions

**Q: "Why split by user and not by chat session?"**
A: Splitting by session means the same person could see both versions on different days. That confuses them and muddies the results.

**Q: "A user submits, comes back, and sees 'you already have a pending request.' Is that a problem?"**
A: No. They were already counted in their group the first time. That message is normal for both groups.

**Q: "What if the card doesn't work on some screens or apps?"**
A: Either leave those screens out of the test for *both* groups, or show those users a backup text version but still count them in the new group. What I can't do is quietly drop them from just one group; that makes the comparison unfair.

---

### Step 5 — Is the Data Clean?

**Goal:** Make sure we can trust the data before looking at results.

**Before launch:** Check that the two groups look alike on credit score, account age, credit limit, and past bot use.

**During the test:**
- **Are the groups close to 50/50?** If not, something is broken.
- **Daily health check:** Card errors, slow responses, agent transfers.

**Don't stop early:** If you keep checking and stop the moment it looks good, you'll often "find" wins that aren't real. Pick the end date upfront and stick to it.

**Say it simply:** "I check the data is healthy first, and I don't call a winner before the planned end date."

#### Follow-Up Questions

**Q: "On Day 4 you have 6,000 users in the old version and 5,400 in the new one. What do you do?"**

A: A gap that big isn't random; something is broken.

1. **Confirm it:** A quick statistical test (chi-square) says a gap this big is almost impossible by chance.
2. **Pause the test** so we stop collecting bad data.
3. **Find the bug with engineering.** Common causes:
   - The card fails to load on some phones or apps, so those users are never recorded.
   - The card takes too long to load and the bot shows an error instead.
   - The "pending request" check works differently in the new version.
4. **Fix it and restart from Day 1.** Don't try to patch the old data; we don't know who went missing.

*If they ask why it matters:* "It's like a store testing a heavy new door that some shoppers can't open. If you only count people who got inside, the new door looks great, but only because it kept people out."

**Q: "Could the new version affect the old group?"**
A: Yes. Both groups share the same agents. If the new version sends more people to agents, wait times go up for everyone. I'd keep an eye on overall agent wait times.

---

### Step 6 — Results

**Goal:** Decide if it worked and check nothing broke.

**What we're testing:**
- **Starting assumption:** The card and the link have the same submission rate.
- **What we want to find out:** Are they different?

**Example result:**
- Old version: **25.0%** submit. New version: **26.8%** submit. (~19,500 users each)
- That's about a **7% lift**.
- **p-value is very small** (well under 0.05) → the lift is real, not luck.
- **Likely range of the lift:** about 4% to 11%. Share this range, not just the p-value; it tells people how big the effect probably is.

**Money:** 26.8% vs. 25% on 500,000 users → ~9,000 more submissions × 40% approved × $100 ≈ **$360K/year**.

**"Do no harm" (guardrail) check:** Agent transfers, repeat questions, approvals, and errors all stayed within agreed limits.

**Breakdowns:** Look at results by app vs. web chat, new vs. long-time customers, and credit score group. Use these for ideas, not final proof.

**Say it simply:** "Submissions went from 25% to 26.8%, about 7% better, it's clearly real, it's worth about $360K a year, and nothing got worse."

#### Follow-Up Questions

**Q: "You looked at 10 breakdowns and one shows a huge lift. Launch just there?"**
A: Not yet. If you look at 10 groups, one will often look great just by luck. I'd use a stricter cutoff or confirm it with a follow-up test.

**Q: "Agent transfers showed no significant change. Does that mean it's safe?"**
A: It means we didn't see harm, but a small increase could still be hiding. I'd check the worst case in the likely range and make sure it's within what we agreed is acceptable.

**Q: "Clicks went up 40% but submissions only 7%. What does that tell you?"**
A: The button gets people to click, but many drop off in the form. So the next thing to improve is the form, not the bot message.

---

### Step 7 — Decision and Launch

**Goal:** Make a clear call and launch safely.

**Recommendation:** Launch. Submissions went up, it's worth ~$360K/year, and nothing got worse.

**Launch gradually:**
- 10% → 50% → 100% of users
- Watch agent transfers and card errors at each step

**Keep a small comparison group:** Leave ~1–5% of users on the old link for a few months to confirm the lift lasts and approvals stay healthy.

**Say it simply:** "Launch gradually and keep a small group on the old version to make sure the win lasts."

#### Follow-Up Questions

**Q: "Submissions went up 7%, but agent transfers rose from 10% to 10.8% because users think the request will hurt their credit score. Do you launch?"**

*(Fits best if the card didn't include the soft-check line, or users missed it.)*

A: I wouldn't rush to launch or cancel. I'd weigh the gain against how easy the problem is to fix.

1. **Compare in dollars:** ~4,000 extra agent chats × $8 ≈ **$32K/year**, vs. **$360K/year** gained. Still clearly worth it.
2. **This confusion is easy to fix:** We use a soft credit check, so there's **no** effect on their score. Users are asking an agent something the bot could have told them.
3. **Read the agent chats:** Is it really about credit score, or something else, like the button not working? Is it one group or one app?
4. **Options:**
   - **Fix and retest (best option):** Make the soft-check line clear and easy to see, right above the button. Add a quick-reply button, "Will this affect my credit score?", so the bot answers it instantly. Run a short follow-up test.
   - **Launch where it works:** If the problem is in one app or group, launch elsewhere first.
   - **Launch and plan staffing:** If the fix will take a while, launch and give the agent team a heads-up.

**Q: "The lift is only 2%, below your 5% target, but the p-value is 0.02. Do you launch?"**

A: This is about "real" vs. "big enough to matter."

1. **What it means:** The p-value says the lift is real. It doesn't say it's big enough to matter. The 5% target was for planning how many users we need, not a pass/fail line.
2. **Quick check:** With our planned users, a 2% lift usually wouldn't show as significant. So I'd ask if the test ran much longer or had more traffic than planned.
3. **Decide:** 2% lift ≈ **$100K/year**.
   - **Cheap and safe** (card is built, nothing got worse): **Launch.** A small real win with no downside is worth taking.
   - **Costly or risky** (hard to maintain, or more agent transfers): **Don't launch yet**; improve it first.

**Q: "The results were flat. What next?"**
A: Check each step. Did clicks go up but submissions not? Did the card show properly everywhere? A flat result still teaches us something, like the real problem being the form and not the bot message.

---

## Glossary (simple definitions)

| Term | What it means |
|---|---|
| A/B test | Show two versions to two random groups and compare |
| Control / Treatment | Old version / new version |
| Main metric | The one number that decides if the test wins |
| "Do no harm" metric (guardrail metric) | A number that must not get worse (e.g., agent transfers) |
| False alarm risk (alpha) | Chance of saying the new version works when it doesn't; usually 5% |
| Chance of catching a real win (power) | Chance of spotting a real lift if it exists; usually 80% |
| Lift | How much better the new version did (e.g., 25% → 26.8% is a ~7% lift) |
| Smallest lift worth finding (MDE) | The smallest improvement worth launching; used to plan the number of users |
| p-value | How likely we'd see this result by luck if there were no real difference. Small = real effect |
| Likely range (confidence interval) | The range the true lift probably falls in |
| Soft credit check | A credit check that doesn't affect the customer's credit score |
| Agent transfer | The bot hands the chat to a live person |
| Pending request | The customer already asked for an increase and is waiting for a decision |
