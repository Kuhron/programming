# idea for trying to see if neural nets can create some language on their own
# game: group of agents are sitting around a table, there is a box with pictures
# agents take turns being "it", the one who looks into the box and sees a picture and must describe it
# neural net takes picture input and produces frequency spectrum output (list of intensities at different frequencies)
# could instead have a list of articulatory positions which are then passed through an agent-specific device to convert it to sound (like mouths, which are similar across people but have some variation which is endowed, not learned)
# either way, the agent converts the picture input to a sound-representation output (but not sound itself)
# the environment then creates sound from this and adds some noise
# other agents receive the sound, it must be passed through a sound-to-representation device (also subject to agent-specific variation in endowment such as ears which have slightly different shapes)
# listeners use neural net to convert the sound representation into a picture, their prediction for what's in the box
# then everyone looks at the picture and updates based on how right/wrong they were
# TODO: somehow introduce time variation so the spectrum of the sound can vary over time, don't want just a single spectrum for whole images to be learned; want time-sequenced language like in real life; maybe if there is a vector of frequencies or basis-spectra (I like basis spectra better, i.e. output cell 1 gives you this spectrum times its activation, output cell 2 gives you a different spectrum, etc., they add together in the sound conversion device to create the output sound, this might also allow for things like many different ways to articulate similar acoustic features; so maybe actually there WOULD be learning from heard sound to which articulations can produce it; each agent could even practice on their own to learn this, like babbling to learn what sounds come out)
# so each agent needs four components: neural net picture -> sound representation; neural net sound representation -> picture (both are trained separately, can't "invert" the function of a neural net by passing things backwards because it won't be one-to-one); device converting sound representation -> sound; device converting sound -> sound representation
# the devices for sound conversion cannot be trained on data; they are just endowed, and have some innate inter-speaker variation
# could they eventually learn to create a conventional system of sound encodings for pictures? i.e. a language to describe the pictures in the box# could even have different agents with different neural net structures
# also maybe they should not evaluate actual similarity of the pictures, but perceived similarity (i.e., would they describe it the same way, they ignore little details that would contribute to the cost function but which they don't care about)
# start simple with these ideas, give them the basic components needed and see what they can learn

