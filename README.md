# 2Q48

This is an experimental mashup of components from the
[2048](https://github.com/gabrielecirulli/2048) game, a successful
[AI](http://ov3y.github.io/2048-AI/) that was developed for it and
[convnetjs](http://cs.stanford.edu/people/karpathy/convnetjs/), a Deep Learning
exploration and visualization JS library. The goal is to try learning how to
play the game with [Q-learning](http://en.wikipedia.org/wiki/Q_learning), with
the added twist that the action-value function is implemented with a neural
network. For the moment it doesn't really learn anything, and to be honest I'm
not even sure that it completely makes sense. To run it, with Node.js:

    $ node js/main.js
