var deepqlearn = require('./convnetjs/deepqlearn');
var Grid = require('./grid')

function GameManager(size, InputManager, Actuator) {
    this.size         = size; // Size of the grid
    
    //var num_inputs = 16; // 9 eyes, each sees 3 numbers (wall, green, red thing proximity)
    var num_inputs = 16 * 11;
    var num_actions = 4; // 5 possible angles agent can turn
    var temporal_window = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
    var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

    // the value function network computes a value of taking any of the possible actions
    // given an input state. Here we specify one explicitly the hard way
    // but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
    // to just insert simple relu hidden layers.
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
    //layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 200, activation:'relu'});
    layer_defs.push({type:'regression', num_neurons:num_actions});

    // options for the Temporal Difference learner that trains the above net
    // by backpropping the temporal difference learning rule.
    var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

    var opt = {};
    opt.temporal_window = temporal_window;
    opt.experience_size = 50; //30000;
    opt.start_learn_threshold = 10;//1000;
    opt.gamma = 0.7;
    opt.learning_steps_total = 200000;
    opt.learning_steps_burnin = 0; //3000;
    opt.epsilon_min = 0.05;
    opt.epsilon_test_time = 0.05;
    opt.layer_defs = layer_defs;
    opt.tdtrainer_options = tdtrainer_options;
    
    var brain = new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo
    this.brain = brain

    this.setup();
    this.run();    
}

// Set up the game
GameManager.prototype.setup = function () {
  this.grid         = new Grid(this.size);
  this.grid.addStartTiles();
  this.score        = 0;
  this.over         = false;
  this.won          = false;

};

// makes a given move and updates state
GameManager.prototype.move = function(direction) {
    
    var prev_smoothness = this.grid.smoothness();
    var prev_occupancy = this.grid.occupancy();
    
    var result = this.grid.move(direction);
    this.score += result.score;
    
    var is_illegal = false;
    if (!result.won) {
        if (result.moved) {
            this.grid.computerMove();
        } else {
            is_illegal = true;
        }
    } else {
        this.won = true;
    }
    
    if (!this.grid.movesAvailable()) {
        this.over = true; // Game over!
    }
    
    return !is_illegal;        
}

// moves continuously until game is over
GameManager.prototype.run = function() {

    console.log('learning while playing..');
    
    while (!this.over && !this.won) {
        
        var action = this.brain.forward(this.grid.getAsNNInputOneHot());            
        var prev_smoothness = this.grid.smoothness();
        var prev_occupancy = this.grid.occupancy();
        var reward = -1;
                
        if (this.move(action)) {
            
            var curr_smoothness = this.grid.smoothness();
            var curr_occupancy = this.grid.occupancy();
            var smoothness_reward = 0;
            if (curr_smoothness < prev_smoothness) {
                smoothness_reward = 1;
            } else if (curr_smoothness > prev_smoothness) {
                smoothness_reward = -1;
            }
            var occupancy_reward = 0;
            if (curr_occupancy <= prev_occupancy) {
                occupancy_reward = 1;
            } else {
                occupancy_reward = -1;
            }
            //console.log(smoothness_reward, occupancy_reward);
            reward = (0.5 * smoothness_reward) + (0.5 * occupancy_reward);
        }
        //console.log(reward);
        this.brain.backward(reward);
        
    }
    console.log('average Q-learning loss: ' + this.brain.average_loss_window.get_average());
    console.log('smooth-ish reward: ' + this.brain.average_reward_window.get_average());
    console.log('best tile:', this.grid.getBestTile());
    console.log('---------------------------------');
    this.setup();
    this.run();    
}

module.exports.GameManager = GameManager;
