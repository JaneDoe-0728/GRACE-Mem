"""Recording what retrieval did, without taking part in it.

    trace   stage waterfall, per-stage add/drop, pass1/pass2 overlap metrics

Pure transformation over results that are already settled: it reads no config,
touches no store, and no retrieval decision depends on it. That is what lets it
sit outside the pipeline that produced the numbers it formats.
"""
