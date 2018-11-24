# playing with the idea of implementing a VM that behaves like a human brain
# especially for purpose of mental calculation
# for example, human may have memorized times tables up to 10 * 10 by heart, along with a few other well-known multiples (15 * 4)
# others have to be computed, which uses up working memory, and their results further stored in short-term memory for rest of calculation

# phenomena to emulate:
# - working memory is limited and degrades over time
# - instead of assigning variables, what really happens is more like associations between things
# -- e.g. last place I parked --> lawrence & kenmore
# -- but overwriting these associations does not get rid of the previous memory; it appends to a history
# -- memory may erroneously retrieve an item that was reinforced more, rather than the most recent value
# -- so working memory is a weighted directed graph, where weights are increased by reinforcement and decreased with disuse



