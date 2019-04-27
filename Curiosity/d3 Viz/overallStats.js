
function plot_scatter(eps){

    // Read Csv files
    d3.text("feature_data_final_int2.csv", function(text) {
        var fdata = d3.csvParseRows(text).map(function(row) {
          return row.map(function(value) {
            return +value;
          });
        });
        
        d3.text("rewards_final_int2.csv", function(text) {
            var data = d3.csvParseRows(text).map(function(row) {
              return row.map(function(value) {
                return +value;
              });
            });
    
            d3.text("feature_data_final.csv", function(text) {
                var fdata1 = d3.csvParseRows(text).map(function(row) {
                  return row.map(function(value) {
                    return +value;
                  });
                });
                
                d3.text("rewards_final.csv", function(text) {
                    var data1 = d3.csvParseRows(text).map(function(row) {
                      return row.map(function(value) {
                        return +value;
                      });
                    });

    
                    d3.select("#overallStatisticView svg").remove();
                    d3.select("#feature_scatter svg").remove();


                    var svg = d3.select('#overallStatisticView').append('svg').attr('width', "100%").attr('height', "10%");
                    
                    // Add texture - taken from https://github.com/riccardoscalco/textures
                    var texture = textures
                    .lines()
                    .heavier(0.7)
                    .stroke("#3F3F3F")
                    .strokeWidth(4)
                    .thicker(1);
    
                    svg.call(texture);
    
                    svg.append('rect')
                    .attr("class", "texturev2")
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("rx", 5)
                    .attr("height", "35%")
                    .attr("width", "100%")
                    .attr("fill", "#4F4F4F");
    
                    svg.append('rect')
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("rx", 5)
                    .attr("height", "35%")
                    .attr("width", "100%")
                    .attr("fill", texture.url());

                    svg.append('text')
                    .attr("x", d3.select('.texturev2').node().getBoundingClientRect().width * (42/100))
                    .attr("y", 20)
                    .text("Detail View : Episode - " + eps)
                    .style("fill","white" )
                    .style("font-size", d3.select('.texturev2').node().getBoundingClientRect().width * (2/100)+"px");
    
                    // Get height and width with respect to the container so that it is not dependent on the screen resolution and the spatial orientation is constant.
                    var width = d3.select('#feature_scatter').node().getBoundingClientRect().width * (94/100);
                    var height = d3.select('#feature_scatter').node().getBoundingClientRect().height * (94/100);
                    var margin = d3.select('#feature_scatter').node().getBoundingClientRect().width * (10/100);
                    var margin_y = d3.select('#feature_scatter').node().getBoundingClientRect().height * (7/100);

                    var svg_container = d3.select("#feature_scatter").append("svg")
                    .attr("width", (width+margin)+"px")
                    .attr("height", (height + margin_y)+"px")
                    .append('g')
                    .attr("transform", `translate(${margin}, ${margin_y})`);

                    // Get our derived Data
                    var points = []
                    var j = fdata[eps].length/2;
                    for(i=0; i<fdata[eps].length/2; i++){
                        points.push({x : fdata[eps][i], y: fdata[eps][j], reward: +data[eps][i]+1})
                        j+=1;
                    }


                    var eps1 = eps;
    
                    var points1 = []
                    var j = fdata1[eps1].length/2;
                    for(i=0; i<fdata1[eps1].length/2; i++){
                        points1.push({x : fdata1[eps1][i], y: fdata1[eps1][j], reward: +data1[eps1][i]+1})
                        j+=1;
                    }
    
                    // Uncomment the following two lines to sort the points w.r.t rewards. 

                    // points1.sort(function(a,b){ return d3.ascending(a.reward, b.reward)})    
                    // points.sort(function(a,b){ return d3.ascending(a.reward, b.reward)})                  
              
                    // Get max and min values of points for scaling purpose.
                    min_x1 = d3.min(points, function(d){ return d.x})
                    max_x1 = d3.max(points, function(d){ return d.x})
    
                    min_y1 = d3.min(points, function(d){ return d.y})
                    max_y1 = d3.max(points, function(d){ return d.y})
    
                    min_x2 = d3.min(points1, function(d){ return d.x})
                    max_x2 = d3.max(points1, function(d){ return d.x})
    
                    min_y2 = d3.min(points1, function(d){ return d.y})
                    max_y2 = d3.max(points1, function(d){ return d.y})
                    
                    min_x = Math.min(min_x1, min_x2)
                    max_x = Math.max(max_x1, max_x2)
    
                    min_y = Math.min(min_y1, min_y2)
                    max_y = Math.max(max_y1, max_y2)
    
                    // console.log(min_x1, max_x1, min_y1, max_y1)
                    // console.log(min_x2, max_x2, min_y2, max_y2)
    
                    //define scales
                    var xScale = d3.scaleLinear()
                    .domain([min_x - 20, max_x + 20])
                    .range([0, width-margin]);
    
                    var yScale = d3.scaleLinear()
                    .domain([min_y - 20, max_y + 20])
                    .range([height-margin_y, 0]);
    
                    //define axis
                    var xAxis = d3.axisBottom(xScale).ticks(10);
                    var yAxis = d3.axisLeft(yScale).ticks(10);
                    
                    svg_container.append("g")
                    .attr("class", "x axis")
                    .attr("transform", `translate(0, ${height-margin_y})`)
                    .call(xAxis);
    
                    svg_container.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                    .attr("y", 15)
                    // .attr("transform", "rotate(-90)")
                    // .attr("fill", "#000");
    
                    // Color scale
                    var color = d3.scaleSequential(d3.interpolateBlues).domain([0,7]);
                    var color1 = d3.scaleSequential(d3.interpolateReds).domain([0,7]);
    
                    // var color = d3.scaleOrdinal()
                    // .domain([1,4,7])
                    // .range([ "#440154ff", "#21908dff", "#fde725ff"])
                    // // Add dots
                     
                    // Draw the scatter plots
                    svg_container.append('g')
                    .selectAll("dot")
                    .data(points)
                    .enter()
                    .append("circle")
                        .attr("cx", function (d) { return xScale(d.x); } )
                        .attr("cy", function (d) { return yScale(d.y); } )
                        .attr("r", 5)
                        .style("fill", function (d) { return color(d.reward); } )
    
                    
                    svg_container.append('g')
                    .selectAll("dot1")
                    .data(points1)
                    .enter()
                    .append("circle")
                        .attr("cx", function (d) { return xScale(d.x); } )
                        .attr("cy", function (d) { return yScale(d.y); } )
                        .attr("r", 5)
                        .style("fill", function (d) { return color1(d.reward); } )


                });
            });
        });
    
    });


}