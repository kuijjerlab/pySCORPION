# Generate R reference outputs for scorpion/tests/test_edge_testing.py.
#
# Uses the installed R SCORPION package and data/ref_networks_df.csv.
# Run from this directory:  Rscript generate_edge_test_data.R

suppressPackageStartupMessages(library(SCORPION))

cat("SCORPION", as.character(packageVersion("SCORPION")), "\n")

out <- function(df, name) {
  write.csv(df, file.path("data", name), row.names = FALSE)
}
quiet <- function(expr) suppressMessages(expr)

nets <- read.csv(file.path("data", "ref_networks_df.csv"), check.names = FALSE,
                 stringsAsFactors = FALSE)
T_cols <- grep("--T$", colnames(nets), value = TRUE)
N_cols <- grep("--N$", colnames(nets), value = TRUE)
B_cols <- grep("--B$", colnames(nets), value = TRUE)

# testEdges ------------------------------------------------------------------
out(quiet(testEdges(nets, "single", group1 = T_cols)), "ref_testEdges_single.csv")
out(quiet(testEdges(nets, "single", group1 = T_cols, empiricalNull = FALSE)),
    "ref_testEdges_single_noemp.csv")
out(quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = N_cols)),
    "ref_testEdges_two_sample.csv")
out(quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = N_cols,
                    empiricalNull = FALSE)),
    "ref_testEdges_two_sample_noemp.csv")
out(quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = N_cols,
                    moderateVariance = FALSE, empiricalNull = FALSE)),
    "ref_testEdges_nomod_noemp.csv")
out(quiet(testEdges(nets, "two.sample", group1 = c("P1--T", "P2--T", "P3--T"),
                    group2 = c("P1--N", "P2--N", "P3--N"), paired = TRUE)),
    "ref_testEdges_paired.csv")
out(quiet(testEdges(nets, "two.sample", group1 = c("P1--T", "P2--T", "P3--T"),
                    group2 = c("P1--N", "P2--N", "P3--N"), paired = TRUE,
                    empiricalNull = FALSE)),
    "ref_testEdges_paired_noemp.csv")
out(quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = N_cols,
                    minLog2FC = 0.3, alternative = "greater",
                    padjustMethod = "holm")),
    "ref_testEdges_two_sample_filtered.csv")
# Parallel path: s0 is computed once over all edges
out(quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = N_cols,
                    minLog2FC = 0.3, nCores = 2, batchSize = 70)),
    "ref_testEdges_two_sample_parallel.csv")

# regressEdges ---------------------------------------------------------------
ordered <- list(Normal = sort(N_cols), Border = sort(B_cols), Tumor = sort(T_cols))
out(regressEdges(nets, ordered), "ref_regressEdges.csv")
out(regressEdges(nets, ordered, padjustMethod = "BY", minMeanEdge = 0.1),
    "ref_regressEdges_BY.csv")

# p.adjust -------------------------------------------------------------------
set.seed(7)
p <- c(runif(40)^3, NA, 0.5, 0.5, 1)
padj <- data.frame(p = p)
for (m in c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "none")) {
  padj[[m]] <- p.adjust(p, method = m)
}
out(padj, "ref_p_adjust.csv")

# maEdges --------------------------------------------------------------------
# Three "studies" from different contrasts; study 2 has missing/invalid SEs
# and study 3 covers only a subset of edges.
s1 <- quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = N_cols))
s2 <- quiet(testEdges(nets, "two.sample", group1 = B_cols, group2 = N_cols))
s3 <- quiet(testEdges(nets, "two.sample", group1 = T_cols, group2 = B_cols))
s2$SE[c(1, 5, 9)] <- NA
s2$SE[c(2, 6)] <- 0
s3 <- s3[seq(1, nrow(s3), by = 2), ]
out(s1, "ma_study1.csv")
out(s2, "ma_study2.csv")
out(s3, "ma_study3.csv")
studies <- list(s1, s2, s3)
out(quiet(maEdges(studies, method = "random")), "ref_maEdges_random.csv")
out(quiet(maEdges(studies, method = "fixed")), "ref_maEdges_fixed.csv")
out(quiet(maEdges(studies, method = "random", minStudies = 3,
                  moderateVariance = FALSE)),
    "ref_maEdges_random_k3_nomod.csv")
out(quiet(maEdges(studies, method = "fixed", s0 = 0.05, padjustMethod = "bonferroni")),
    "ref_maEdges_fixed_s0.csv")
