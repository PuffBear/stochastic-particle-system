# Interview Guide: Coordination Diagnostic Validation

**Purpose:** Determine whether multi-robot and multi-sensor engineering teams encounter the specific failure mode the diagnostic targets — deploying communication without being able to verify whether it is helping or hurting.

**Target:** 8–10 engineers across robotics, drone systems, and sensor networks. At least 2 from each of: (a) warehouse/industrial robotics, (b) field robotics (agriculture, search-and-rescue, environmental), (c) sensor network engineering.

**Success gate:** ≥3 interviewees provide a documented case (not hypothetical) where they could not determine whether their communication protocol was helping or hurting team performance.

---

## Sourcing interviewees

**ROS community:**
- Search GitHub for repos tagged `multi-robot`, `swarm-robotics`, `multi-agent` with >50 stars and recent commits
- ROS Discourse threads about coordination failures or communication overhead
- ROSCon attendee lists (public) — filter for roles containing "systems", "autonomy", "field robotics"

**Agricultural drones:**
- Sentera, PrecisionHawk, DJI Enterprise developer community
- AUVSI Xponential conference attendees
- University precision agriculture programmes (Purdue, UC Davis, Wageningen)

**Search-and-rescue / field robotics:**
- DARPA Subterranean Challenge team alumni (team lists are public)
- IEEE Safety, Security, and Rescue Robotics conference authors

**Sensor networks:**
- USGS and NOAA instrument network engineers (LinkedIn, agency directories)
- Environmental IoT companies (Campbell Scientific, Onset, In-Situ)

**Warehouse robotics:**
- Mid-size integrators (not Amazon/Kiva — they build in-house and won't talk)
- 6 River Systems, Locus Robotics, Fetch Robotics alumni
- MHI (Material Handling Institute) member companies

---

## Interview structure (45 minutes)

### Opening (5 min)
Introduce the research context briefly — studying multi-agent coordination in robotics and sensing systems, specifically how teams validate their communication protocols.

Do not describe the tool yet. Let the interviewee surface problems unprompted before showing them a solution.

### Block 1: Current system (10 min)

1. "Walk me through how your system's agents currently communicate with each other. What do they share, how often, and how does it affect their behaviour?"

2. "How did you decide what to put in those messages? Was it designed upfront, or did it evolve?"

3. "How do you currently know if the communication is working — what tells you it's contributing positively?"

### Block 2: Failure and uncertainty (15 min)

4. "Has there ever been a situation where you suspected the communication might be hurting rather than helping — either slowing things down, causing synchronisation problems, or making decisions correlated in a bad way?"
   - If yes: "What happened? How did you figure out what was going on?"
   - If no: "Is that something you've ever wondered about?"

5. "If you changed the communication protocol tomorrow — swapped in different message content — how confident are you that you'd be able to tell whether the new version was better or worse?"

6. "Have you ever run an A/B test or a controlled comparison on your communication protocol — same task, same conditions, with and without messages, or with different message content?"
   - If yes: "How did you set that up? What was hard about it?"
   - If no: "What would it take to run one?"

### Block 3: The diagnostic (10 min)

*Now describe the three-condition test at a high level:*

"One approach is to run the same scenarios three ways: once with your actual messages, once with the messages randomly scrambled so each agent gets a real message but not the one addressed to it, and once with no messages at all. The gap between scrambled and nothing tells you how much the channel structure helps; the gap between actual and scrambled tells you if the specific content matters."

7. "Does that kind of test seem feasible in your simulation environment? What would be hard to set up?"

8. "If this produced a report showing those three numbers — actual vs scrambled vs nothing — would that be useful? Who would you show it to?"

9. "Is there a specific decision you've faced, or expect to face, where having that report would have changed what you did?"

### Closing (5 min)

10. "Is there anyone else you'd suggest I talk to about this — either someone who's run into this problem or someone who's solved it in a way you haven't?"

---

## Logging

For each interview, record:
- Date, role, company type (not name unless consented), system type
- Q4 answer: documented case / speculative / no
- Q5 answer: confidence level (high/medium/low) and reasoning
- Q6 answer: has run / hasn't run / wouldn't know how
- Q8 answer: useful / not useful / conditional
- Q9 answer: specific decision / general interest / none
- Q10: referrals

**Documented case criterion:** The interviewee describes a specific past situation (not a hypothetical) where they did not know whether their communication protocol was helping or hurting, and could not determine this with their available tools.

---

## What to do if the gate fails

If <3 documented cases emerge after 10 interviews:

1. Check whether interviewees understood the question. Restate Q4 more concretely: "Has a communication-related bug or misbehaviour ever taken more than a day to diagnose?" This lowers the bar without changing the underlying problem.

2. Expand to simulation engineers at game companies or digital-twin teams — the failure mode is not unique to physical robots.

3. Check secondary sources: GitHub issues in multi-robot repos mentioning "communication" + "degraded performance" or "turned off messages"; Stack Overflow questions about multi-agent debugging.

4. If still no documented cases after expanded search: the market exists only where teams have already solved this problem differently. Document why the gate failed and close this path.

Do not proceed to prototype integration or paid engagements without the gate passing.
