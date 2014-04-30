var deepqlearn = require('./convnetjs/deepqlearn');
var Grid = require('./grid')

function GameManager(size) {
    this.size         = size; // Size of the grid

    // one-hot: 16 * 11 inputs
    // normal: 16 inputs
    this.use_one_hot_board_encoding = true;
    var num_inputs = 16 * (this.use_one_hot_encoding ? 11 : 1);
    var num_actions = 4; // 5 possible angles agent can turn
    var temporal_window = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
    var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

    var layer_defs = [];
    // I have no idea what's the right topology for this..
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'regression', num_neurons:num_actions});

    // options for the Temporal Difference learner that trains the above net
    // by backpropping the temporal difference learning rule.
    var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

    var opt = {};
    opt.temporal_window = temporal_window;
    opt.experience_size = 100; //30000;
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

        var input_arr = this.use_one_hot_board_encoding ?
                           this.grid.getAsNNInputOneHot() : this.grid.getAsNNInput();
        var action = this.brain.forward(input_arr);
        var prev_smoothness = this.grid.smoothness();
        var prev_occupancy = this.grid.occupancy();
        var prev_monotonicity = this.grid.monotonicity();
        var reward = -1; // illegal move (i.e. could not move in that direction) has neg reward
                
        if (this.move(action)) { // legal
            
            var curr_smoothness = this.grid.smoothness();
            var curr_occupancy = this.grid.occupancy();
            var curr_monotonicity = this.grid.monotonicity();
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
            var monotonicity_reward = 0;
            if (curr_monotonicity > prev_monotonicity) {
                monotonicity_reward = 1;
            } else if (curr_monotonicity < prev_monotonicity) {
                monotonicity_reward = -1;
            }
            //console.log(smoothness_reward, occupancy_reward);
            reward = (1/3 * smoothness_reward) + (1/3 * occupancy_reward) + (1/3 * monotonicity_reward);
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
