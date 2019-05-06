function plot_dist1(eps){
        
    d3.select("#ep_dist1 svg").remove();

    d3.text("viz_data/actions_final12.csv", function(text) {
        var adata = d3.csvParseRows(text).map(function(row) {
        return row.map(function(value) {
            return +value;
        });
        });
        
        d3.text("viz_data/rewards_final12.csv", function(text) {
            var rdata = d3.csvParseRows(text).map(function(row) {
            return row.map(function(value) {
                return +value;
            });
            });

            // // console.log(rdata[eps][0])

            var width = d3.select('#ep_dist1').node().getBoundingClientRect().width * (94/100);
            var height = d3.select('#ep_dist1').node().getBoundingClientRect().height * (94/100);
            var margin = d3.select('#ep_dist1').node().getBoundingClientRect().width * (10/100);
            var margin_y = d3.select('#ep_dist1').node().getBoundingClientRect().height * (7/100);

            var svg_container = d3.select("#ep_dist1").append("svg")
                    .attr("width", (width+margin)+"px")
                    .attr("height", (height + margin_y)+"px")
                    .append('g')
                    .attr("transform", `translate(${margin}, ${margin_y})`);
            var points = []
            var j = 0;
            for(i=0; i<adata[eps].length; i++){
                points.push({x : adata[eps][i], y: i, reward: rdata[eps][j]});
                j +=1;

            }
            // // console.log(points)
            min_x1 = d3.min(points, function(d){ return d.x})
            max_x1 = d3.max(points, function(d){ return d.x})

            min_y1 = d3.min(points, function(d){ return d.y})
            max_y1 = d3.max(points, function(d){ return d.y})

            // // console.log(min_x1, min_y1, max_x1, max_y1);

            var xScale = d3.scaleLinear()
            .domain([min_x1-1 , max_x1+1 ])
            .range([0, width-margin]);

            var yScale = d3.scaleLinear()
            .domain([min_y1-1 , max_y1+1 ])
            .range([height-margin_y, 0]);

            var xAxis = d3.axisBottom(xScale).ticks(3);
            var yAxis = d3.axisLeft(yScale).ticks(10);

            svg_container.append("g")
            .attr("class", "x axis")
            .attr("transform", `translate(0, ${height-margin_y})`)
            .call(xAxis);

            svg_container.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .attr("y", 15)
            actions = [{"action":"fire"},{"action":"left"},{"action":"right"}];

            var color = d3.scaleOrdinal().domain(actions).range(["blue", "green", "orange"]);

            // var color = d3.scaleSequential(d3.interpolateReds).domain([min_x1,(max_x1+1)*5]);
            // // console.log(points)
            svg_container.append('g')
                .selectAll("dot")
                .data(points)
                .enter()
                .append("circle")
                    .attr("cx", function (d) { return xScale(d.x); } )
                    .attr("cy", function (d) { return yScale(d.y); } )
                    .attr("r", 4)
                    .attr("fill-opacity", function(d) { return d.reward/10 + 0.1;})
                    .style("fill", function (d) { return color(d.x); } )


        });

    });
}