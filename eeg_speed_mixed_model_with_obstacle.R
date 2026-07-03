# Speed mixed model for EEG gait data (age x light + obstacle).

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
# SPEED
# =========================================================

speed_df <- read_csv("eeg_gait_speed_with_obstacle_final.csv")

# Exclude PID 49
speed_df <- speed_df[speed_df$pid != "PID 49", ]

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

# Test unequal variance by light AND obstacle combined
m_speed_nlme_varLightObs <- lme(
  speed ~ age * light + obstacle,
  random = ~1 | pid,
  weights = varIdent(form = ~1 | light * obstacle),
  data = speed_df
)

# Compare against varLight only
anova(m_speed_nlme_varLight, m_speed_nlme_varLightObs)

# Also compare varLight vs base+varObs to see which matters more
AIC(m_speed_nlme_base, m_speed_nlme_varLight, m_speed_nlme_varObs, m_speed_nlme_varLightObs)


# =========================================================
# FINAL MODEL: nlme with varIdent by light
# =========================================================

summary(m_speed_nlme_varLight)
anova(m_speed_nlme_varLight)


# =========================================================
# EFFECT SIZES 
# =========================================================

eta2p_age <- (21.962 * 1) / (21.962 * 1 + 38)
cat("eta2p age:", eta2p_age, "\n")

eta2p_light <- (41.798 * 2) / (41.798 * 2 + 433)
cat("eta2p light:", eta2p_light, "\n")

eta2p_obstacle <- (33.064 * 3) / (33.064 * 3 + 433)
cat("eta2p obstacle:", eta2p_obstacle, "\n")

eta2p_agelight <- (0.466 * 2) / (0.466 * 2 + 433)
cat("eta2p age:light:", eta2p_agelight, "\n")

# =========================================================
# ESTIMATED MARGINAL MEANS AND POST-HOCS
# =========================================================

# Age
emmeans(m_speed_nlme_varLight, ~ age)

# Light pairwise comparisons
emmeans(m_speed_nlme_varLight, pairwise ~ light, adjust = "bonferroni")


# =========================================================
# CUSTOM OBSTACLE CONTRASTS (matching EEG post-hoc structure)
# These use the same estimated marginal means from the final model.
# Obstacle levels in order: expected_absent, expected_present, 
#                           unexpected_absent, unexpected_present
# =========================================================

emm_obs_custom <- emmeans(m_speed_nlme_varLight, ~ obstacle)

# 5 planned contrasts matching EEG post-hoc structure, Bonferroni corrected
# Obstacle levels: 1=expected_absent, 2=expected_present, 
#                  3=unexpected_absent, 4=unexpected_present
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
# Save collapsed means
write.csv(as.data.frame(collapsed_means), "eeg_speed_obstacle_collapsed_means.csv", row.names = FALSE)

# ANOVA results + eta squared
anova_res <- as.data.frame(anova(m_speed_nlme_varLight))
anova_res$Effect <- rownames(anova_res)
anova_res$eta2p <- c(NA, eta2p_age, eta2p_light, eta2p_obstacle, eta2p_agelight)
write.csv(anova_res, "eeg_speed_obstacle_anova_results.csv", row.names = FALSE)

# EMM for age
emm_age <- as.data.frame(emmeans(m_speed_nlme_varLight, ~ age))
write.csv(emm_age, "eeg_speed_obstacle_emm_age.csv", row.names = FALSE)

# EMM pairwise for light
emm_light <- emmeans(m_speed_nlme_varLight, pairwise ~ light, adjust = "bonferroni")
write.csv(as.data.frame(emm_light$emmeans), "eeg_speed_obstacle_emm_light_means.csv", row.names = FALSE)
write.csv(as.data.frame(emm_light$contrasts), "eeg_speed_obstacle_emm_light_contrasts.csv", row.names = FALSE)

# Save obstacle means
write.csv(as.data.frame(emm_obs_custom), "eeg_speed_obstacle_emm_obstacle_means.csv", row.names = FALSE)

# Save custom contrasts to CSV
custom_contrasts_df <- as.data.frame(custom_contrasts)
write.csv(custom_contrasts_df, "eeg_speed_obstacle_custom_contrasts.csv", row.names = FALSE)