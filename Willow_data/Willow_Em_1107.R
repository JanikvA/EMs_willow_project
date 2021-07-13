library(ggplot2)
library(GGally)
library(ggExtra)

#set working directory

#read in data ----

#Mai 2021

Lb1_3 <- read.table("LB1_3.csv", header=T, sep=";")
Lb2_3 <- read.table("LB2_3.csv", header=T, sep=";")
Lb5_3 <- read.csv("LB5_3.csv", header=T, sep=";")
Sb1_3 <- read.csv("SB1_3.csv", header=T, sep=";")
Sb3_3 <- read.csv("SB3_3.csv", header=T, sep=";")
Sb4_3 <- read.csv("SB4_3.csv", header=T, sep=";")
Sb5_3 <- read.csv("SB5_3.csv", header=T, sep=";")

#2020

Lb1_2 <- read.table("LB1_2.csv", header=T, sep=";")
Lb2_2 <- read.table("LB2_2.csv", header=T, sep=";")
Lb5_2 <- read.csv("LB5_2.csv", header=T, sep=";")
Sb1_2 <- read.csv("SB1_2.csv", header=T, sep=";")
Sb3_2 <- read.csv("SB3_2.csv", header=T, sep=";")
Sb4_2 <- read.csv("SB4_2.csv", header=T, sep=";")
Sb5_2 <- read.csv("SB5_2.csv", header=T, sep=";")

#create dataframes for both sand and loam ----

#2021
Sb_3 <- merge(Sb1_3, Sb3_3, all=T)
Sb_3 <- merge(Sb_3, Sb4_3, all=T)
Sb_3 <- merge(Sb_3, Sb5_3, all=T)

Sb_3$Height <- as.numeric(sub(",", ".", Sb_3$Height, fixed = TRUE))
Sb_3$Diameter <- as.numeric(sub(",", ".", Sb_3$Diameter, fixed = TRUE))


Lb_3 <- merge(Lb1_3, Lb2_3, all=T)
Lb_3 <- merge(Lb_3, Lb5_3, all=T)

Lb_3$Height <- as.numeric(sub(",", ".", Lb_3$Height, fixed = TRUE))
Lb_3$Diameter <- as.numeric(sub(",", ".", Lb_3$Diameter, fixed = TRUE))

#2020

Sb_2 <- merge(Sb1_2, Sb3_2, all=T)
Sb_2 <- merge(Sb_2, Sb4_2, all=T)
Sb_2 <- merge(Sb_2, Sb5_2, all=T)

Sb_2$Height <- as.numeric(sub(",", ".", Sb_2$Height, fixed = TRUE))
Sb_2$Diameter <- as.numeric(sub(",", ".", Sb_2$Diameter, fixed = TRUE))


Lb_2 <- merge(Lb1_2, Lb2_2, all=T)
Lb_2 <- merge(Lb_2, Lb5_2, all=T)

Lb_2$Height <- as.numeric(sub(",", ".", Lb_2$Height, fixed = TRUE))
Lb_2$Diameter <- as.numeric(sub(",", ".", Lb_2$Diameter, fixed = TRUE))

###Compare sand and loam samples ----
Lb_3$Soiltype= "Loam"
Sb_3$Soiltype= "Sand"
Comb_LS21<- merge(Lb_3, Sb_3, all=T)

# boxplot: Height and Diameter by soil type 
ggplot(data=Comb_LS21,aes(x=Soiltype, y=Height))+
  geom_boxplot()
#Willows in Loam seem to be larger

ggplot(data=Comb_LS21,aes(x=Soiltype, y=Diameter))+
  geom_boxplot()

#Scatterplot: Height and Diameter

ggplot(Comb_LS21, aes(x=Height, y=Diameter, color=Soiltype))+
  geom_point()


hist(Lb_3$Height) #approximately normally distributed
hist(Lb_3$Diameter) #not normally distributed
hist(Sb_3$Height) #not normally distributed
hist(Sb_3$Diameter) #not normally distributed

#We will use a Mann-Whitney-U-Test because there is no normal distribution

wilcox.test(Lb_3$Height, Sb_3$Height, alternative="two.sided") #significantly different
wilcox.test(Lb_3$Diameter, Sb_3$Diameter, alternative="two.sided") #significantly different

#Both Height and Diameter are significantly higher in loam


#Compare 2020 and 2021 ----
Lb_2$Year="20"
Lb_3$Year="21"
Comb_L20_21<- merge(Lb_2, Lb_3, all=T)

Sb_2$Year="20"
Sb_3$Year="21"
Comb_S20_21<- merge(Sb_2, Sb_3, all=T)

#Boxplots- Diameter
ggplot(data=Comb_L20_21,aes(x=Year, y=Diameter))+
  geom_boxplot()  #nearly the same
ggplot(data=Comb_L20_21,aes(x=Year, y=Height))+
  geom_boxplot()  #

#Boxplots- Height
ggplot(data=Comb_S20_21,aes(x=Year, y=Diameter))+
  geom_boxplot()
ggplot(data=Comb_S20_21,aes(x=Year, y=Height))+
  geom_boxplot()

hist(Lb_2$Height) #not normally distributed
hist(Lb_2$Diameter) #not normally distributed
hist(Sb_2$Height) #not normally distributed
hist(Sb_2$Diameter) #not normally distributed

#We will use a Mann-Whitney-U-Test because there is no normal distribution

wilcox.test(Lb_2$Height, Lb_3$Height, alternative="two.sided") #significant difference
wilcox.test(Lb_2$Diameter, Lb_3$Diameter, alternative="two.sided") #no significant difference

wilcox.test(Sb_2$Height, Sb_3$Height, alternative="two.sided") #significant larger in 2021
wilcox.test(Sb_2$Diameter, Sb_3$Diameter, alternative="two.sided") #no significant difference

# the correlation between height and diameter 
#Based on the soil type, if the ratio is the same or not
#Loam
L_20=na.omit(Lb_2)
S_20=na.omit(Sb_2)
cor(L_20$Height, L_20$Diameter)
cor(S_20$Height, S_20$Diameter)

#2021
L_21=na.omit(Lb_3)
S_21=na.omit(Sb_3)
cor(L_21$Height, L_21$Diameter)
cor(S_21$Height, S_21$Diameter)

# add another column with the ratio of Height/ Diameter and compare them
# -> would be interesting! if the proportion of height and diameter change, 
# how the willows adjust with the environment changes