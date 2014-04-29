var gm = require('./game_manager');

animationDelay = 100;
minSearchTime = 100;

// Wait till the browser is ready to render the game (avoids glitches)
//window.requestAnimationFrame(function () {
var manager = new gm.GameManager(4); //, KeyboardInputManager, HTMLActuator);
//});
