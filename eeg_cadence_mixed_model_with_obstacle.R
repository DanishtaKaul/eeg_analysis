# Cadence mixed model for EEG gait data (age x light + obstacle).

library(readr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(effectsize)
library(performance)
library(see)
library(car)
library(nlme)
# =========================================================
# CADENCE
# =========================================================

cadence_df <- read_csv("eeg_gait_cadence_with_obstacle_final.csv")

# Exclude PID 49
cadence_df <- cadence_df[cadence_df$pid != "PID 49", ]

# Check structure
str(cadence_df)

# Convert to factors
cadence_df$pid <- factor(cadence_df$pid)
cadence_df$age <- factor(cadence_df$age)
cadence_df$light <- factor(cadence_df$light)
cadence_df$obstacle <- factor(cadence_df$obstacle)

# Set reference levels
cadence_df$age <- relevel(cadence_df$age, ref = "young")
cadence_df$light <- relevel(cadence_df$light, ref = "light")

levels(cadence_df$age)
levels(cadence_df$light)
levels(cadence_df$obstacle)

# =========================================================
# FIT CADENCE MODEL: age x light + obstacle main effect
# =========================================================

m_cadence <- lmer(cadence ~ age * light + obstacle + (1 | pid), data = cadence_df)

summary(m_cadence)
anova(m_cadence)

# =========================================================
# ASSUMPTION CHECKS
# =========================================================

# 1 Posterior predictive check
dev.new()
check_model(m_cadence, check = "pp_check")

# 2 Linearity
dev.new()
check_model(m_cadence, check = "linearity")

# 3 Homogeneity of variance
dev.new()
check_model(m_cadence, check = "homogeneity")

# 4 Influential observations
dev.new()
check_model(m_cadence, check = "outliers")

# 5 Collinearity
dev.new()
check_model(m_cadence, check = "vif")

# 6 Normality of residuals
dev.new()
check_model(m_cadence, check = "normality")

# 7 Normality of random effects
dev.new()
check_model(m_cadence, check = "reqq")

# =========================================================
# VARIANCE STRUCTURE TESTS (nlme)
# =========================================================

# Base nlme model
m_cadence_nlme_base <- lme(
  cadence ~ age * light + obstacle,
  random = ~1 | pid,
  data = cadence_df
)

# Test unequal variance by light
m_cadence_nlme_varLight <- lme(
  cadence ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | light),
  data = cadence_df
)

anova(m_cadence_nlme_base, m_cadence_nlme_varLight)

# Test unequal variance by age
m_cadence_nlme_varAge <- lme(
  cadence ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | age),
  data = cadence_df
)

anova(m_cadence_nlme_base, m_cadence_nlme_varAge)

# Test unequal variance by obstacle
m_cadence_nlme_varObs <- lme(
  cadence ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | obstacle),
  data = cadence_df
)

anova(m_cadence_nlme_base, m_cadence_nlme_varObs)

# Test combined light x obstacle variance structure
m_cadence_nlme_varLightObs <- lme(
  cadence ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | light * obstacle),
  data = cadence_df
)

# Compare against varLight only
anova(m_cadence_nlme_varLight, m_cadence_nlme_varLightObs)

# Compare AICs
AIC(m_cadence_nlme_base, m_cadence_nlme_varLight, m_cadence_nlme_varObs, m_cadence_nlme_varLightObs)

# =========================================================
# FINAL MODEL: nlme with varIdent by light
# =========================================================

summary(m_cadence_nlme_varLight)
anova(m_cadence_nlme_varLight)

# =========================================================
# EFFECT SIZES 
# =========================================================

eta2p_age <- (6.945 * 1) / (6.945 * 1 + 38)
cat("eta2p age:", eta2p_age, "\n")

eta2p_light <- (19.455 * 2) / (19.455 * 2 + 433)
cat("eta2p light:", eta2p_light, "\n")

eta2p_obstacle <- (27.086 * 3) / (27.086 * 3 + 433)
cat("eta2p obstacle:", eta2p_obstacle, "\n")

eta2p_agelight <- (2.448 * 2) / (2.448 * 2 + 433)
cat("eta2p age:light:", eta2p_agelight, "\n")

# =========================================================
# ESTIMATED MARGINAL MEANS AND POST-HOCS
# =========================================================

# Age
emmeans(m_cadence_nlme_varLight, ~ age)

# Light pairwise
emmeans(m_cadence_nlme_varLight, pairwise ~ light, adjust = "bonferroni")


# =========================================================
# CUSTOM OBSTACLE CONTRASTS (matching EEG post-hoc structure)
# Obstacle levels: 1=expected_absent, 2=expected_present, 
#                  3=unexpected_absent, 4=unexpected_present
# =========================================================

emm_obs_custom <- emmeans(m_cadence_nlme_varLight, ~ obstacle)

# 5 planned contrasts matching EEG post-hoc structure, Bonferroni corrected
custom_contrasts <- contrast(emm_obs_custom, list(
  "present - absent" = c(-0.5, 0.5, -0.5, 0.5),
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0),
  "UP - absent" = c(-0.5, 0, -0.5, 1),
  "EP - absent" = c(-0.5, 1, -0.5, 0)
), adjust = "bonferroni")

custom_contrasts

# Get collapsed means for present and absent (with SE)
collapsed_means <- contrast(emm_obs_custom, list(
  "absent" = c(0.5, 0, 0.5, 0),
  "present" = c(0, 0.5, 0, 0.5)
))
collapsed_means
# =========================================================
# EXPORT RESULTS TO CSV
# =========================================================

# ANOVA results + eta squared
anova_res <- as.data.frame(anova(m_cadence_nlme_varLight))
anova_res$Effect <- rownames(anova_res)
anova_res$eta2p <- c(NA, eta2p_age, eta2p_light, eta2p_obstacle, eta2p_agelight)
write.csv(anova_res, "eeg_cadence_obstacle_anova_results.csv", row.names = FALSE)

# EMM for age
emm_age <- as.data.frame(emmeans(m_cadence_nlme_varLight, ~ age))
write.csv(emm_age, "eeg_cadence_obstacle_emm_age.csv", row.names = FALSE)

# EMM pairwise for light
emm_light <- emmeans(m_cadence_nlme_varLight, pairwise ~ light, adjust = "bonferroni")
write.csv(as.data.frame(emm_light$emmeans), "eeg_cadence_obstacle_emm_light_means.csv", row.names = FALSE)
write.csv(as.data.frame(emm_light$contrasts), "eeg_cadence_obstacle_emm_light_contrasts.csv", row.names = FALSE)

# Save obstacle means
write.csv(as.data.frame(emm_obs_custom), "eeg_cadence_obstacle_emm_obstacle_means.csv", row.names = FALSE)

# Save collapsed means
write.csv(as.data.frame(collapsed_means), "eeg_cadence_obstacle_collapsed_means.csv", row.names = FALSE)
# Save custom contrasts
custom_contrasts_df <- as.data.frame(custom_contrasts)
write.csv(custom_contrasts_df, "eeg_cadence_obstacle_custom_contrasts.csv", row.names = FALSE)