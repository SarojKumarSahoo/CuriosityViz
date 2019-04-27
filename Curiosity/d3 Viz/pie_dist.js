    d3.text("actions_final_int2.csv", function(text) {
        // Read csv
        var data = d3.csvParseRows(text).map(function(row) {
        return row.map(function(value) {
            return +value;
        });
        });

        d3.text("actions_final.csv", function(text) {
            var data1 = d3.csvParseRows(text).map(function(row) {
            return row.map(function(value) {
                return +value;
            });
            });

            d3.text("rewards_final_int2.csv", function(text) {
                var rdata = d3.csvParseRows(text).map(function(row) {
                return row.map(function(value) {
                    return +value;
                });
                });

                d3.text("rewards_final.csv", function(text) {
                    var rdata1 = d3.csvParseRows(text).map(function(row) {
                    return row.map(function(value) {
                        return +value;
                    });
                    });


                    var counts = {};
                    // Derived data for vis
                    for (var i = 0; i < data.length; i++) {
                        for (var ii = 0; ii < data[i].length; ii++) {
                            counts[data[i][ii]] = 1 + (counts[data[i][ii]] || 0);
                        }
                    }

                    // Get distribution count for action
                    data = [{"action": "fire", "value": counts[0]},
                            {"action": "right", "value": counts[1]},
                            {"action": "left", "value": counts[2]}]


                    // console.log(Math.min(counts[0], counts[1], counts[2]))

                    var width = d3.select('#ep_rw_dist').node().getBoundingClientRect().width * (94/100);
                    var height = d3.select('#ep_rw_dist').node().getBoundingClientRect().height * (94/100);
                    var margin = d3.select('#ep_rw_dist').node().getBoundingClientRect().width * (10/100);
                    var margin_y = d3.select('#ep_rw_dist').node().getBoundingClientRect().height * (7/100);
                    var radius = Math.min(width, height) / 4;

                    var color = d3.scaleSequential(d3.interpolateBlues).domain([Math.min(counts[0], counts[1], counts[2]), Math.max(counts[0], counts[1], counts[2])]);
                    
                    // create our pie chart
                    var arc = d3.arc()
                        .outerRadius(radius - 10)
                        .innerRadius(0);

                    var labelArc = d3.arc()
                        .outerRadius(radius - 40)
                        .innerRadius(radius - 40);

                    var pie = d3.pie()
                        .sort(null)
                        .value(function(d) { return d.value; });

                    var svg = d3.select("#ep_rw_dist").append("svg")
                        .attr("width", (width+margin)+"px")
                        .attr("height", (height + margin_y)+"px")
                        .append('g')
                        .attr("transform", "translate(" + width / 4 + "," + height / 4 + ")");

                    var g = svg.selectAll(".arc")
                        .data(pie(data))
                        .enter().append("g")
                        .attr("class", "arc");

                    g.append("path")
                        .attr("d", arc)
                        .style("fill", function(d) { return color(d.value); });

                    g.append("text")
                        .attr("transform", function(d) { return "translate(" + labelArc.centroid(d) + ")"; })
                        .text(function(d,i) { return d.data["action"]; })
                        .style("font-size", "10px");




                    var counts1 = {};

                    for (var i = 0; i < data1.length; i++) {
                        for (var ii = 0; ii < data1[i].length; ii++) {
                            counts1[data1[i][ii]] = 1 + (counts1[data1[i][ii]] || 0);
                        }
                    }

                    data1 = [{"action": "fire", "value": counts1[0]},
                            {"action": "right", "value": counts1[1]},
                            {"action": "left", "value": counts1[2]}]
                    var color1 = d3.scaleSequential(d3.interpolateReds).domain([Math.min(counts1[0], counts1[1], counts1[2]), Math.max(counts1[0], counts1[1], counts1[2])]);


                    var svg = d3.select("#ep_rw_dist").select("svg")
                        .attr("width", (width+margin)+"px")
                        .attr("height", (height + margin_y)+"px")
                        .append('g')
                        .attr("transform", "translate(" + width / 4 + "," + 3*height / 4 + ")");

                    var g = svg.selectAll(".arc")
                        .data(pie(data1))
                        .enter().append("g")
                        .attr("class", "arc");

                    g.append("path")
                        .attr("d", arc)
                        .style("fill", function(d) { return color1(d.value); });

                    g.append("text")
                        .attr("transform", function(d) { return "translate(" + labelArc.centroid(d) + ")"; })
                        .attr("text-anchor", "middle")
                        .text(function(d,i) { return d.data["action"]; })
                        .style("font-size", "10px");


                        var rcounts = {};

                        for (var i = 0; i < rdata.length; i++) {
                            for (var ii = 0; ii < rdata[i].length; ii++) {
                                rcounts[rdata[i][ii]] = 1 + (rcounts[rdata[i][ii]] || 0);
                            }
                        }
                        
                        rdata = [{"reward": "0", "value": rcounts[0]||0},
                        {"reward": "1", "value": rcounts[1]||0},
                        {"reward": "4", "value": rcounts[4]||0}]
                        var color2 = d3.scaleSequential(d3.interpolateBlues).domain([Math.min(rcounts[0], rcounts[1], rcounts[4]), Math.max(rcounts[0], rcounts[1], rcounts[4])]);

                        
                            
                        // console.log(rdata)
                        
        
                        var svg = d3.select("#ep_rw_dist").select("svg")
                            .attr("width", (width+margin)+"px")
                            .attr("height", (height + margin_y)+"px")
                            .append('g')
                            .attr("transform", "translate(" + 3*width / 4 + "," + height / 4 + ")");
        
                        var g = svg.selectAll(".arc")
                            .data(pie(rdata))
                            .enter().append("g")
                            .attr("class", "arc");
        
                        g.append("path")
                            .attr("d", arc)
                            .style("fill", function(d) { return color2(d.value); });
        
                        g.append("text")
                            .attr("transform", function(d) { return "translate(" + labelArc.centroid(d) + ")"; })
                            .text(function(d) { return d.data["reward"]; })
                            .style("font-size", "10px");

                        


                        var rcounts1 = {};

                        for (var i = 0; i < rdata1.length; i++) {
                            for (var ii = 0; ii < rdata1[i].length; ii++) {
                                rcounts1[rdata1[i][ii]] = 1 + (rcounts1[rdata1[i][ii]] || 0);
                            }
                        }
                        
                        rdata1 = [{"reward": "0", "value": rcounts1[0]||0},
                        {"reward": "1", "value": rcounts1[1]||0},
                        {"reward": "4", "value": rcounts1[4]||0}, 
                        {"reward": "7", "value": rcounts1[7]||0}]
                        var color3 = d3.scaleSequential(d3.interpolateReds).domain([Math.min(rcounts1[0], rcounts1[1], rcounts1[4], rcounts1[7]), Math.max(rcounts1[0], rcounts1[1], rcounts1[4], rcounts1[7])]);

                        
                        // console.log(rdata1)
                        
        
                        var svg = d3.select("#ep_rw_dist").select("svg")
                            .attr("width", (width+margin)+"px")
                            .attr("height", (height + margin_y)+"px")
                            .append('g')
                            .attr("transform", "translate(" + 3*width / 4 + "," + 3*height / 4 + ")");
        
                        var g = svg.selectAll(".arc")
                            .data(pie(rdata1))
                            .enter().append("g")
                            .attr("class", "arc");
        
                        g.append("path")
                            .attr("d", arc)
                            .style("fill", function(d) { return color3(d.value); });
        
                        g.append("text")
                            .attr("transform", function(d) { return "translate(" + labelArc.centroid(d) + ")"; })
                            .text(function(d) { return d.data["reward"]; })
                            .style("font-size", "10px");

                        });

            });

        });



    });
