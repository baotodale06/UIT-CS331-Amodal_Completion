# Completed annotated Surface-PMC benchmark

Both full runs are verified and registered. Full LPIPS improves **1.191%**
[0.749%, 1.641%], with **4.733×** allocation time. The original 204-target
primary cohort remains inconclusive, and the earlier 3% full target is unmet.
No new training, tuning or inference was launched for this review.

- [Four-page report](REPORT.pdf)
- [Report with evidence links](REPORT.md)
- [All 2,565 paired target records](paired-records.csv)
- [Independent local audit](audit.json)
- [Report validation](validation.json) and [visual inspection](visual-review.json)
- [Job accounting and registry recovery](remote-evidence/receipt.json)

The portable ZIP includes the original 16,197-member metadata archive, independent
SHA-256 receipts, exact paired values, selector replay, frozen parameters, and
analysis source. Full native PNGs remain on the cluster; neural metrics were not
rerun locally. The original primary-cohort PDF is preserved under `evidence/`.

There is no remaining job to resume. Future tuning requires a new design and
independent evaluation; these completed outcomes are now inspected data.
