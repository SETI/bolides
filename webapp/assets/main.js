function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

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

// observer to wait for a change to the orbit-map tab and plot orbits
// when that happens
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
otherTabObserver()

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

// Create a skybox using NASA TYCHO artwork.
//const skybox = viz.createStars();

//const skybox = viz.createSkybox({
//  textureUrl: '{{assets}}/skybox/nasa_tycho.jpg'
//});

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

    //if (e>0.95){
    //    return;
    //}
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
