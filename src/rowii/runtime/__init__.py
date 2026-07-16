"""Runtime prototype (Step-2 package 6): persistable monitor artifacts.

`rowii.runtime.snapshot` bundles everything the `monitor` CLI needs to label and
score a NEW recording -- fitted state detector + per-state scoring references +
conformal thresholds -- into one pickle-free `MonitorSnapshot` artifact: the
serialization point `rowii.state.detect.FittedDetector` deliberately deferred
("not serialized -- future runtime-prototype serialization point", package-2 spec
D1; realized here per package-6 spec D1).
"""
