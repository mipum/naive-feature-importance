# naive-feature-importance
A straightforward approach to evaluate feature importance, applied to Keras

Follows a question asked by @fchollet at https://twitter.com/fchollet/status/688591631319085056

The proposed approach is to create a generator modifying test input by zeroing one feature at a time across all sampples, and to observe impact of modified data on model accuracy vs. accuracy achieved on a complete test input
