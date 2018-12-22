# coding: utf-8
# Ethics in Data Science
# Considerations:
#
# **Starting point: you are incredibly fortunate to be a student in MSAN, to be a data scientist in San Francisco. You have a responsibility to not do harm with your skills.**
# **Not everything your employer asks you to do may be legal.**  An engineer at Volkswagen was [sentenced to 3.5 years in prison](https://www.nytimes.com/2017/08/25/business/volkswagen-engineer-prison-diesel-cheating.html) for helping develop the software to cheat on federal emissions tests.  Your boss asking you to do something is not an excuse that will protect you in court.
# **Not everything that is legal is ethical.**
# **What questions are worth asking?**
# **What data should be collected?** (Privacy, de-anonymization, risks of leaks.) -
# Unintended consequences: **Impact can be very different from intent.** E.g. impact of online (semi-automated) censorship of violent videos.
# **Don't blindly optimize for a metric (or group of metrics) without thinking about bigger picture.**  As [Abe Gong points out](https://www.youtube.com/watch?list=PLB2SCq-tZtVmadnKpO8WwKiFKteY5rHPT&time_continue=1624&v=WjKdKvDS10g), including information about whether someone's father left when they were a child could increase accuracy of a model for criminial recidivism, but is it moral to do so?
#
# <img src="images/ethics_recidivism.jpg" alt="digit" style="width: 40%"/>
#   (Source: [Ethics for Powerful Algorithms](https://www.youtube.com/watch?list=PLB2SCq-tZtVmadnKpO8WwKiFKteY5rHPT&time_continue=1624&v=WjKdKvDS10g))
# **Costs of Mistakes**
# *There’s no hiding behind algorithms anymore.* -Atlantic reporter Alexis Madrigal.  After the recent shooting in Vegas, [Google surfaced a 4chan story](https://www.theatlantic.com/technology/archive/2017/10/google-and-facebook-have-failed-us/541794/) as a top search result and Facebook's Trending Stories showed a page from a known source of Russian propaganda.
# **Self-reinforcing Feedback loops**
#
# Eg. if you don't show women tech meetups, then you'll get even less women coming to tech meetups, which will cause the algorithm to suggest even less tech meetups to women, etc... See Evan Estola: [When Recommendation Systems Go Bad](https://www.youtube.com/watch?v=MqoRzNhrTnQ)
#
# Sending more cops to predominantly Black neighborhoods will increase reported black crime, increasing bias against blacks in policing algorithms.
