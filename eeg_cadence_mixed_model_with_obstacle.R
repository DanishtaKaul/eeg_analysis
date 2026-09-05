# Cadence mixed model for EEG gait data (age x light + obstacle)
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
# CADENCE WITH OBSTACLE
# =========================================================

cadence_df <- read_csv("eeg_gait_cadence_with_obstacle_final.csv")


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

anova(m_cadence_nlme_varObs, m_cadence_nlme_varLightObs)

AIC(m_cadence_nlme_base, m_cadence_nlme_varLight, m_cadence_nlme_varObs, m_cadence_nlme_varLightObs)


# =========================================================
# Final variance structure: varIdent by obstacle
# =========================================================

summary(m_cadence_nlme_varObs)
anova(m_cadence_nlme_varObs)   # supplies F-values and DenDF for partial eta-squared below

# =========================================================
# Partial eta-squared
# =========================================================

eta2p_age       <- (2.778  * 1) / (2.778  * 1 + 39)
eta2p_light     <- (12.071 * 2) / (12.071 * 2 + 444)
eta2p_obstacle  <- (35.573 * 3) / (35.573 * 3 + 444)
eta2p_agelight  <- (2.140  * 2) / (2.140  * 2 + 444)

cat("eta2p age:      ", round(eta2p_age, 4),      "\n")
cat("eta2p light:    ", round(eta2p_light, 4),    "\n")
cat("eta2p obstacle: ", round(eta2p_obstacle, 4), "\n")
cat("eta2p age:light:", round(eta2p_agelight, 4), "\n")

# Age main effect 
emmeans(m_cadence_nlme_varObs, ~ age)

# Light: estimated means + pairwise contrasts, Bonferroni-corrected
emmeans(m_cadence_nlme_varObs, pairwise ~ light, adjust = "bonferroni")

# =========================================================
# Obstacle contrasts (planned, matching EEG post-hoc structure)
# Levels: 1=expected_absent, 2=expected_present,
#         3=unexpected_absent, 4=unexpected_present
# =========================================================

emm_obs_custom <- emmeans(m_cadence_nlme_varObs, ~ obstacle)

# 5 planned contrasts, Bonferroni-corrected
custom_contrasts <- contrast(emm_obs_custom, list(
  "present - absent" = c(-0.5, 0.5, -0.5, 0.5),
  "UP - EP"          = c(0, -1, 0, 1),
  "UA - EA"          = c(-1, 0, 1, 0),
  "UP - absent"      = c(-0.5, 0, -0.5, 1),
  "EP - absent"      = c(-0.5, 1, -0.5, 0)
), adjust = "bonferroni")
custom_contrasts

# Collapsed present/absent means (with SE)
collapsed_means <- contrast(emm_obs_custom, list(
  "absent"  = c(0.5, 0, 0.5, 0),
  "present" = c(0, 0.5, 0, 0.5)
))
collapsed_means

# Export results to CSV - all from final model m_cadence_nlme_varObs

anova_res <- as.data.frame(anova(m_cadence_nlme_varObs))
anova_res$Effect <- rownames(anova_res)
anova_res$eta2p  <- c(NA, eta2p_age, eta2p_light, eta2p_obstacle, eta2p_agelight)
write.csv(anova_res, "eeg_cadence_obstacle_anova_results.csv", row.names = FALSE)

emm_age <- as.data.frame(emmeans(m_cadence_nlme_varObs, ~ age))
write.csv(emm_age, "eeg_cadence_obstacle_emm_age.csv", row.names = FALSE)

emm_light <- emmeans(m_cadence_nlme_varObs, pairwise ~ light, adjust = "bonferroni")
write.csv(as.data.frame(emm_light$emmeans),   "eeg_cadence_obstacle_emm_light_means.csv",     row.names = FALSE)
write.csv(as.data.frame(emm_light$contrasts), "eeg_cadence_obstacle_emm_light_contrasts.csv", row.names = FALSE)

write.csv(as.data.frame(emm_obs_custom),   "eeg_cadence_obstacle_emm_obstacle_means.csv", row.names = FALSE)
write.csv(as.data.frame(collapsed_means),  "eeg_cadence_obstacle_collapsed_means.csv",    row.names = FALSE)
write.csv(as.data.frame(custom_contrasts), "eeg_cadence_obstacle_custom_contrasts.csv",   row.names = FALSE)
