d3.text("actions_1.csv", function(text) {
    var data = d3.csvParseRows(text).map(function(row) {
        return row.map(function(value) {
        return value;
        });
    });

    d3.text("rewards_1.csv", function(text) {
        var r_data = d3.csvParseRows(text).map(function(row) {
            return row.map(function(value) {
            return value;
            });
        });

        test(data, r_data, 0)

        })
    })


function get_graph_1(n){
    console.log(n)
    d3.selectAll("#sp1 svg").remove();

    d3.text("actions_1.csv", function(text) {
        var data = d3.csvParseRows(text).map(function(row) {
            return row.map(function(value) {
            return value;
            });
        });

        d3.text("rewards_1.csv", function(text) {
            var r_data = d3.csvParseRows(text).map(function(row) {
                return row.map(function(value) {
                return value;
                });
            });

            test(data, r_data, n)

            })
        })

}    
    

    
function test(data, r_data,n){
    var width = 500,
    height = 500,
    start = 0,
    end = 2.25,
    numSpirals = 4;

    var step_num;
    var color = d3.scaleOrdinal(['lightblue', 'lightgreen', 'cyan', 'steelblue']);

    var theta = function(r) {
        return numSpirals * Math.PI * r;
    };

    var r = d3.min([width, height]) / 2 - 40;

    var radius = d3.scaleLinear()
        .domain([start, end])
        .range([40, r]);

    var svg1 = d3.select("#sp1").append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

    // create the spiral, borrowed from http://bl.ocks.org/syntagmatic/3543186
    var points = d3.range(start, end + 0.001, (end - start) / 1000);

    var spiral = d3.radialLine()
        .curve(d3.curveCardinal)
        .angle(theta)
        .radius(radius);

    var path = svg1.append("path")
        .datum(points)
        .attr("id", "spiral")
        .attr("d", spiral)
        .style("fill", "none")
        .style("stroke", "steelblue");

    var spiralLength = path.node().getTotalLength(),
        N = data[n].length,
        barWidth = (spiralLength / N) - 1;
    var someData = [];
    cat = []
    for (var i = 0; i < N; i++) {
        cat.push(i)
        var currentDate = new Date();
        currentDate.setDate(currentDate.getDate() + i);
        someData.push({
          cat: i,
          action: data[n][i],
          value: +r_data[n][i]+1
      });

    }
    console.log(cat)

    var ordinalScale = d3.scaleBand()
      .domain(cat)
      .range([0, spiralLength]);
    
    var yScale = d3.scaleLinear()
      .domain([0, d3.max(someData, function(d){
        return d.value;
      })])
      .range([0, (r / numSpirals) - 30]);

    svg1.selectAll("rect")
      .data(someData)
      .enter()
      .append("rect")
      .attr("x", function(d,i){
        
        var linePer = ordinalScale(d.cat),
            posOnLine = path.node().getPointAtLength(linePer),
            angleOnLine = path.node().getPointAtLength(linePer - barWidth);
      
        d.linePer = linePer; 
        d.x = posOnLine.x; 
        d.y = posOnLine.y; 
        
        d.a = (Math.atan2(angleOnLine.y, angleOnLine.x) * 180 / Math.PI) - 90; //angle at the spiral position

        return d.x;
      })
      .attr("y", function(d){
        return d.y;
      })
      .attr("width", function(d){
        return barWidth;
      })
      .attr("height", function(d){
        return yScale(d.value);
      })
      .attr("class", function(d,i) { return "pt" + i; })
      .style("fill", function(d){return color(d.action);})
      .style("stroke", "none")
      .attr("transform", function(d){
        return "rotate(" + d.a + "," + d.x  + "," + d.y + ")"; 
      });

      var tooltip = d3.select("#sp1")
    .append('div')
    .attr('class', 'tooltip');

    tooltip.append('div')
    .attr('class', 'action');
    tooltip.append('div')
    .attr('class', 'reward');

    svg1.selectAll("rect")
    .on('mouseover', function(d,i) {

        step_num = d.cat;
        tooltip.select('.action').html("Action: <b>" + d.action + "</b>");
        tooltip.select('.reward').html("Reward: <b>" + d.value + "<b>");

        d3.selectAll("rect.pt" + i)
        .style("fill", "red")


        tooltip.style('display', 'block');
        tooltip.style('opacity',2);
        
        
    })
    .on('mousemove', function(d) {
        tooltip.style('top', (d3.event.layerY + 10) + 'px')
        .style('left', (d3.event.layerX - 25) + 'px');
    })
    .on('mouseout', function(d,i) {
            
            tooltip.style('display', 'none');
            tooltip.style('opacity',0);
            d3.selectAll("rect.pt" + i).style("fill", function(d){return color(d.action);});
        })
    

}

    