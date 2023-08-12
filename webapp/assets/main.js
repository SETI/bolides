// Main javascript file for the orbit map

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/* Function which creates an observer to watch for changes
 * to hidden orbital-elements div updated by the app.
 * When it changes, it calls plotorbits() to update the orbits.
 * It also sets up another observer to watch for changes to which
 * tab is being displayed, and calls otherTabObserver() when that happens.*/
console.log('setting up observer');
function orbitTabObserver(){
    var update_time = 0;

    // use an observer to watch the table of orbital elements
    // (which also contains colors) and update whenever it changes
    var target = document.querySelector('#orbital-elements');
    var optionObserver = new MutationObserver(function (mutations, me) {
        console.log('change observer called')
        // `mutations` is an array of mutations that occurred
        // `me` is the MutationObserver instance
        plotorbits();
        //for (const mutation of mutations) {
            //console.log(mutation)
        //}

    });
    optionObserver.observe(target, {
        attributes: true,
    });

    // observer to call the other tab observer when the tab changes
    var observer = new MutationObserver(function (mutations, me) {
      // `mutations` is an array of mutations that occurred
      // `me` is the MutationObserver instance
      var canvas = document.getElementById('orbit-map');
      if (!canvas) {
        me.disconnect(); // stop observing
        optionObserver.disconnect();
        otherTabObserver();
        return;
      }
    });
    // start observing
    observer.observe(document, {
      childList: true,
      subtree: true
    });
}

/* Observer to wait for the orbit-map element to appear (which will happen
 * when the tab is switchced to Orbits), plot the orbits, and go back to
 * the orbitTabObserver. */
function otherTabObserver(){
    // set up the mutation observer
    var observer = new MutationObserver(function (mutations, me) {
      var canvas = document.getElementById('orbit-map');
      if (canvas) {
        viz = plotorbits();
        me.disconnect(); // stop observing
        orbitTabObserver();
        return;
      }
    });
    // start observing
    observer.observe(document, {
      childList: true,
      subtree: true
    });
}

// as we start on the Earth map tab, start the otherTabObserver.
otherTabObserver()


// function to plot the orbits on orbit-map.
function plotorbits(){

document.getElementById('orbit-map').innerHTML = "";

// Create the visualization and put it in our div.
const viz = new Spacekit.Simulation(document.getElementById('orbit-map'), {
  basePath: '.',
  jd: 2458454.5,
  maxNumParticles: 2 ** 16,
  debug: {
    // showAxesHelper: true,
    showStats: false,
  },
});
console.log('viz created');

// Create our first object - the sun - using a preset space object.
const sun = viz.createObject('sun', Spacekit.SpaceObjectPresets.SUN);

// Then add some planets
viz.createObject('mercury', Spacekit.SpaceObjectPresets.MERCURY);
viz.createObject('venus', Spacekit.SpaceObjectPresets.VENUS);
viz.createObject('earth', Spacekit.SpaceObjectPresets.EARTH);
viz.createObject('mars', Spacekit.SpaceObjectPresets.MARS);
viz.createObject('jupiter', Spacekit.SpaceObjectPresets.JUPITER);
viz.createObject('saturn', Spacekit.SpaceObjectPresets.SATURN);
viz.createObject('uranus', Spacekit.SpaceObjectPresets.URANUS);
viz.createObject('neptune', Spacekit.SpaceObjectPresets.NEPTUNE);

console.log('planets plotted')

// add orbits
data = JSON.parse(document.getElementById('orbital-elements').textContent);
idx = 0;
for(const element of data){
    a = element[0];
    e = element[1];
    q = element[2];
    i = element[3] * Math.PI / 180;
    om = element[4] * Math.PI / 180;
    w = element[5] * Math.PI / 180;
    color = element[6];

    tp = 2454080.665214673185

    n = 0;//Math.pow(Math.abs(a), -3/2)

    // create a Spacekit Ephem object containing the orbital elements
    const ephem = new Spacekit.Ephem({
        a: a,
        e: e,
        q: q,
        i: i,
        om: om,
        w: w,
        ma: 0,
        epoch: Math.random() * 2500000,
        q: 1,
        tp: tp,
        n: n
    });

    // try to add the the ephemeris to the visualization
    try {
        viz.createObject(idx, {
            hideOrbit: false,
            particleSize: 1,
            textureUrl: '{{assets}}/sprites/transparent.png',
            ephem,
            theme: {
                orbitColor: color,
            },
        });
    }
    catch (error) {
        console.error(error);
    }
    idx = idx+1;
};

console.log('orbits plotted');
return viz;
}
