d3.csv("episode_data_final.csv", function(error2, data) {
    d3.csv("episode_data_final_int2.csv", function(error2, data1) {

    
    // console.log(data)
    var svg = d3.select('#episodeStatisticView').append('svg').attr('width', "100%").attr('height', "10%");
    
    // Add texture - taken from https://github.com/riccardoscalco/textures
    var texture = textures
    .lines()
    .heavier(0.7)
    .stroke("#3F3F3F")
    .strokeWidth(4)
    .thicker(1);

    svg.call(texture);

    svg.append('rect')
    .attr("class", "texture")
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
    .attr("x", d3.select('.texture').node().getBoundingClientRect().width * (38/100))
    .attr("y", 20)
    .text("Statistics View")
    .style("fill","white" )
    .style("font-size", d3.select('.texture').node().getBoundingClientRect().width * (3/100)+"px");

    var width = d3.select('#ep_rw_chart').node().getBoundingClientRect().width * (88/100);
    var height = d3.select('#ep_rw_chart').node().getBoundingClientRect().height * (88/100);
    var margin = d3.select('#ep_rw_chart').node().getBoundingClientRect().width * (10/100);
    var margin_y = d3.select('#ep_rw_chart').node().getBoundingClientRect().height * (7/100);

    // var chart_svg = d3.select('#ep_rw_chart').append("svg")
    // .attr("width", width + margin)
    // .attr("height", height + margin_y)
    // .attr("transform", "translate(" + margin + "," + margin_y + ")");
    n1 = data.length
    n2 = data1.length

    n = Math.max(n1,n2);

    // Scale 
    var xScale = d3.scaleLinear()
    .domain([0,n-1])
    .range([0, width-margin]);

    var yScale = d3.scaleLinear()
    .domain([0,150])
    .range([height-margin_y, 0]);

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    // Add SVG 
    var chart_svg = d3.select("#ep_rw_chart").append("svg")
    .attr("x", margin)
    .attr("width", (width+margin)+"px")
    .attr("height", (height + margin_y)+"px")
    .append('g')
    .attr("transform", `translate(${margin}, ${margin_y})`);

    var brush = d3.brushX()                   
        .extent( [ [0,0], [width+margin,height + margin_y] ] )  
        .on("end", updateChart) 

    // console.log(d3.extent([ [0,0], [width+2*margin,height + 2*margin_y]]) )

    var line = d3.line()
        .x(function(d) { return xScale(d.Episode)  }) 
        .y(function(d) { return yScale(d.Reward); }) 
        .curve(d3.curveMonotoneX)
    
    

    // Plot 0th episode
    plot_scatter(0)
    plot_dist1(0);
    plot_dist2(0);

    chart_svg.append("path")
        .attr("class", "line1") 
        .attr("fill", "none")
        .attr("stroke", "orange")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("d", line(data))

    chart_svg.selectAll("linec")
        .data(data)
        .enter()
        .append("circle")
        .attr("class", "data-circle")
        .attr("r", 1)
        .style("fill", "orange")
        .attr("cx", function(d) { return xScale(d.Episode); })
        .attr("cy", function(d) { return yScale(d.Reward); })
        .on("mouseover", function(d){

            d3.select(this).attr("r", 3);

            chart_svg.append("line")
            .attr("class", "mouseover-line")
            .style("stroke", "#000")
            .attr("stroke-opacity", 0.3)
            .attr("x1", xScale(d.Episode))
            .attr("x2", xScale(d.Episode))
            .attr("y1", yScale(0))
            .attr("y2", yScale(150));
        } )
        .on("mouseout", function(d){
            d3.select(this).attr("r", 1);

            d3.selectAll(".mouseover-line").remove();

        })
        .on("click", function(d){
            // console.log(d3.selectAll(".mouse-line"))
            d3.selectAll(".mouse-line").remove();

            chart_svg.append("line")
            .attr("class", "mouse-line")
            .style("stroke", "#000")
            .attr("x1", xScale(d.Episode))
            .attr("x2", xScale(d.Episode))
            .attr("y1", yScale(0))
            .attr("y2", yScale(150));

            plot_scatter(d.Episode);
            plot_dist1(d.Episode);
            plot_dist2(d.Episode);
        })


    // Line chart
    chart_svg.append("path")
        .attr("class", "line2") 
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("d", line(data1))

    // Add circles to line to make it clickable 
    chart_svg.selectAll("linec2")
        .data(data1)
        .enter()
        .append("circle")
        .attr("class", "data-circle")
        .attr("r", 1)
        .style("fill", "steelblue")
        .attr("cx", function(d) { return xScale(d.Episode); })
        .attr("cy", function(d) { return yScale(d.Reward); })
        .on("mouseover", function(d){

            // increase radius and draw line on mouseover
            d3.select(this).attr("r", 3);
            // console.log(d.Episode)
            chart_svg.append("line")
            .attr("class", "mouseover-line")
            .style("stroke", "#000")
            .attr("stroke-opacity", 0.3)
            .attr("x1", xScale(d.Episode))
            .attr("x2", xScale(d.Episode))
            .attr("y1", yScale(0))
            .attr("y2", yScale(150));
            
        } )
        .on("mouseout", function(d){
            // remove line and decrease radius
            d3.select(this).attr("r", 1);
            d3.selectAll(".mouseover-line").remove();
        })
        .on("click", function(d){

            // Make line fixed on click
            d3.selectAll(".mouse-line").remove();

            d3.select(this).attr("r", 3);
            chart_svg.append("line")
            .attr("class", "mouse-line")
            .style("stroke", "#000")
            .attr("x1", xScale(d.Episode))
            .attr("x2", xScale(d.Episode))
            .attr("y1", yScale(0))
            .attr("y2", yScale(150));

            plot_scatter(d.Episode);
            plot_dist1(d.Episode);
            plot_dist2(d.Episode);
        })


    // Brush and zoom feature but somehow can't integrate with on_click

    // chart_svg
    // .append("g")
    //     .attr("class", "brush")
    //     .call(brush);
        
    xAxis = d3.axisBottom(xScale)        
    .ticks(12);
    yAxis = d3.axisLeft(yScale)
    .ticks(12);

    chart_svg.append('g')
    .attr("class", "x_axis")
    .attr("transform", `translate(0, ${height-margin_y})`)
    .call(xAxis);

    chart_svg.append("g")
    .attr("class", "y_axis")
    .call(yAxis)
    .append('text')
    .attr("y", 15)
    .attr("transform", "rotate(-90)")
    .attr("fill", "#000");
    // .text("Reward");  




    function updateChart() {

        extent = d3.event.selection
    
        xScale.domain([ xScale.invert(extent[0]), xScale.invert(extent[1]) ])
        // console.log(xScale.invert(extent[0]), xScale.invert(extent[1]))
        chart_svg.select(".brush").call(brush.move, null) 
        
    

        chart_svg.select(".x_axis").transition().duration(1000)
        .call(d3.axisBottom(xScale));


        chart_svg.select(".line1")
        .transition()
        .duration(1000)
        .attr("d", line(data))

        chart_svg.select(".line2")
        .transition()
        .duration(1000)
        .attr("d", line(data1))

        chart_svg.on("dblclick",function(){
            xScale.domain(d3.extent(data, function(d) { return d.Episode; }))
            chart_svg.select(".x_axis").transition().duration(1000)
            .call(d3.axisBottom(xScale));
            
            chart_svg.select(".line1")
            .transition()
            .duration(1000)
            .attr("d", line(data))

            chart_svg.select(".line2")
            .transition()
            .duration(1000)
            .attr("d", line(data1))
        
        });
        }
    });
});

