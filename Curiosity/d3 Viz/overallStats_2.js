d3.text("feature_data_final12.csv", function(text) {
    var fdata = d3.csvParseRows(text).map(function(row) {
      return row.map(function(value) {
        return +value;
      });
    });
    
    d3.text("rewards_final12.csv", function(text) {
        var data = d3.csvParseRows(text).map(function(row) {
          return row.map(function(value) {
            return +value;
          });
        });

        

        console.log(fdata[0].length);
        console.log(data[0].length);



        var width = d3.select('#feature_scatter').node().getBoundingClientRect().width * (94/100);
        var height = d3.select('#feature_scatter').node().getBoundingClientRect().height * (94/100);
        var margin = d3.select('#feature_scatter').node().getBoundingClientRect().width * (10/100);
        var margin_y = d3.select('#feature_scatter').node().getBoundingClientRect().height * (7/100);

        var svg_container1 = d3.select("#feature_scatter").select("svg")

        var eps = 75;
        var points = []
        var j = fdata[eps].length/2;
        for(i=0; i<fdata[eps].length/2; i++){
            points.push({x : fdata[eps][i], y: fdata[eps][j], reward: +data[eps][i]+1})
            j+=1;
        }

        console.log(points)
        min_x = d3.min(points, function(d){ return d.x})
        max_x = d3.max(points, function(d){ return d.x})

        min_y = d3.min(points, function(d){ return d.y})
        max_y = d3.max(points, function(d){ return d.y})
        

        var xScale = d3.scaleLinear()
        .domain([min_x - 20, max_x + 20])
        .range([0, width-margin]);

        var yScale = d3.scaleLinear()
        .domain([min_y - 20, max_y + 20])
        .range([height-margin_y, 0]);

        // // Color scale: give me a specie name, I return a color
        var color = d3.scaleSequential(d3.interpolateBlues).domain([0,7]);
        // var color = d3.scaleOrdinal()
        // .domain([1,4,7])
        // .range([ "#440154ff", "#21908dff", "#fde725ff"])
        // // Add dots
        svg_container1.append('g')
        .selectAll("dot1")
        .data(points)
        .enter()
        .append("circle")
            .attr("cx", function (d) { return xScale(d.x); } )
            .attr("cy", function (d) { return yScale(d.y); } )
            .attr("r", 5)
            .style("fill", function (d) { return color(d.reward); } )
            });

});