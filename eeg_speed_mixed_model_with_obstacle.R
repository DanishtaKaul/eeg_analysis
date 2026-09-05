# Speed mixed model for EEG gait data (age x light + obstacle)

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
# SPEED WITH OBSTACLE
# =========================================================

speed_df <- read_csv("eeg_gait_speed_with_obstacle_final.csv")


# Check structure
str(speed_df)

# Convert to factors
speed_df$pid <- factor(speed_df$pid)
speed_df$age <- factor(speed_df$age)
speed_df$light <- factor(speed_df$light)
speed_df$obstacle <- factor(speed_df$obstacle)

# Set reference levels
speed_df$age <- relevel(speed_df$age, ref = "young")
speed_df$light <- relevel(speed_df$light, ref = "light")

levels(speed_df$age)
levels(speed_df$light)
levels(speed_df$obstacle)

# =========================================================
# FIT SPEED MODEL: age x light + obstacle main effect
# =========================================================

m_speed <- lmer(speed ~ age * light + obstacle + (1 | pid), data = speed_df)

summary(m_speed)
anova(m_speed)

# =========================================================
# ASSUMPTION CHECKS
# =========================================================

# Residuals
resid_speed <- residuals(m_speed)
fitted_speed <- fitted(m_speed)

# 1 Posterior predictive check
dev.new()
check_model(m_speed, check = "pp_check")

# 2 Linearity
dev.new()
check_model(m_speed, check = "linearity")

# 3 Homogeneity of variance
dev.new()
check_model(m_speed, check = "homogeneity")

# 4 Influential observations
dev.new()
check_model(m_speed, check = "outliers")

# 5 Collinearity
dev.new()
check_model(m_speed, check = "vif")

# 6 Normality of residuals
dev.new()
check_model(m_speed, check = "normality")

# 7 Normality of random effects
dev.new()
check_model(m_speed, check = "reqq")

# =========================================================
# VARIANCE STRUCTURE TESTS (nlme)
# =========================================================

# Base nlme model (no variance structure)
m_speed_nlme_base <- lme(
  speed ~ age * light + obstacle,
  random = ~1 | pid,
  data = speed_df
)

# Test unequal variance by light
m_speed_nlme_varLight <- lme(
  speed ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | light),
  data = speed_df
)

anova(m_speed_nlme_base, m_speed_nlme_varLight)

# Test unequal variance by age
m_speed_nlme_varAge <- lme(
  speed ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | age),
  data = speed_df
)

anova(m_speed_nlme_base, m_speed_nlme_varAge)

# Test unequal variance by obstacle
m_speed_nlme_varObs <- lme(
  speed ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | obstacle),
  data = speed_df
)

anova(m_speed_nlme_base, m_speed_nlme_varObs)

# Test combined light x obstacle variance structure
m_speed_nlme_varLightObs <- lme(
  speed ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | light * obstacle),
  data = speed_df
)

anova(m_speed_nlme_varLight, m_speed_nlme_varLightObs)

AIC(m_speed_nlme_base, m_speed_nlme_varLight, m_speed_nlme_varObs, m_speed_nlme_varLightObs)

# =========================================================
# Final variance structure: varIdent by light
# =========================================================

summary(m_speed_nlme_varLight)
anova(m_speed_nlme_varLight)   # supplies F-values and denDF for partial eta-squared

# =========================================================
# Partial eta-squared
# =========================================================

eta2p_age       <- (16.378 * 1) / (16.378 * 1 + 39)
eta2p_light     <- (47.031 * 2) / (47.031 * 2 + 444)
eta2p_obstacle  <- (32.771 * 3) / (32.771 * 3 + 444)
eta2p_agelight  <- (1.572  * 2) / (1.572  * 2 + 444)

cat("eta2p age:      ", round(eta2p_age, 4),      "\n")
cat("eta2p light:    ", round(eta2p_light, 4),    "\n")
cat("eta2p obstacle: ", round(eta2p_obstacle, 4), "\n")
cat("eta2p age:light:", round(eta2p_agelight, 4), "\n")

# Age main effect 
emmeans(m_speed_nlme_varLight, ~ age)

# Light: estimated means + pairwise contrasts, Bonferroni-corrected
emmeans(m_speed_nlme_varLight, pairwise ~ light, adjust = "bonferroni")


# =========================================================
# Obstacle contrasts (planned, matching EEG post-hoc structure)
# Levels: 1=expected_absent, 2=expected_present,
#         3=unexpected_absent, 4=unexpected_present
# =========================================================

emm_obs_custom <- emmeans(m_speed_nlme_varLight, ~ obstacle)

# 5 planned contrasts, Bonferroni-corrected
custom_contrasts <- contrast(emm_obs_custom, list(
  "present - absent" = c(-0.5, 0.5, -0.5, 0.5),
  "UP - EP"          = c(0, -1, 0, 1),
  "UA - EA"          = c(-1, 0, 1, 0),
  "UP - absent"      = c(-0.5, 0, -0.5, 1),
  "EP - absent"      = c(-0.5, 1, -0.5, 0)
), adjust = "bonferroni")
custom_contrasts

# Collapsed present/absent means for speed (with SE)
collapsed_means <- contrast(emm_obs_custom, list(
  "absent"  = c(0.5, 0, 0.5, 0),
  "present" = c(0, 0.5, 0, 0.5)
))
collapsed_means

# =========================================================
# Export results to CSV - all from final model m_speed_nlme_varLight
# =========================================================

# ANOVA table + partial eta-squared
# eta2p vector maps to rows: (Intercept)=NA, age, light, obstacle, age:light
anova_res <- as.data.frame(anova(m_speed_nlme_varLight))
anova_res$Effect <- rownames(anova_res)
anova_res$eta2p  <- c(NA, eta2p_age, eta2p_light, eta2p_obstacle, eta2p_agelight)
write.csv(anova_res, "eeg_speed_obstacle_anova_results.csv", row.names = FALSE)

# Age EMM
emm_age <- as.data.frame(emmeans(m_speed_nlme_varLight, ~ age))
write.csv(emm_age, "eeg_speed_obstacle_emm_age.csv", row.names = FALSE)

# Light EMM + pairwise contrasts (Bonferroni)
emm_light <- emmeans(m_speed_nlme_varLight, pairwise ~ light, adjust = "bonferroni")
write.csv(as.data.frame(emm_light$emmeans),   "eeg_speed_obstacle_emm_light_means.csv",     row.names = FALSE)
write.csv(as.data.frame(emm_light$contrasts), "eeg_speed_obstacle_emm_light_contrasts.csv", row.names = FALSE)

# Obstacle EMM, collapsed present/absent means, planned contrasts
write.csv(as.data.frame(emm_obs_custom),   "eeg_speed_obstacle_emm_obstacle_means.csv", row.names = FALSE)
write.csv(as.data.frame(collapsed_means),  "eeg_speed_obstacle_collapsed_means.csv",    row.names = FALSE)
write.csv(as.data.frame(custom_contrasts), "eeg_speed_obstacle_custom_contrasts.csv",   row.names = FALSE)
