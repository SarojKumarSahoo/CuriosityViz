d3.csv("episode_data12.csv", function(error2, data) {
    d3.csv("episode_data_final_int.csv", function(error2, data1) {
    console.log(data)
    d3.select("#stck_chart svg").remove();
    
    var width = d3.select('#stck_chart').node().getBoundingClientRect().width * (88/100);
    var height = d3.select('#stck_chart').node().getBoundingClientRect().height * (88/100);
    var margin = d3.select('#stck_chart').node().getBoundingClientRect().width * (10/100);
    var margin_y = d3.select('#stck_chart').node().getBoundingClientRect().height * (7/100);

    // var chart_svg = d3.select('#stck_chart').append("svg")
    // .attr("width", width + margin)
    // .attr("height", height + margin_y)
    // .attr("transform", "translate(" + margin + "," + margin_y + ")");
    n1 = data.length
    n2 = data1.length

    n = Math.max(n1,n2);


    /* Scale */
    var xScale = d3.scaleLinear()
    .domain([0,n-1])
    .range([0, width-margin]);

    var yScale = d3.scaleLinear()
    .domain([0,200])
    .range([height-margin_y, 0]);

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    // /* Add SVG */
    var chart_svg = d3.select("#stck_chart").append("svg")
    .attr("x", margin)
    .attr("width", (width+margin)+"px")
    .attr("height", (height + margin_y)+"px")
    .append('g')
    .attr("transform", `translate(${margin}, ${margin_y})`);

    var brush = d3.brushX()                   
        .extent( [ [0,0], [width+margin,height + margin_y] ] )  
        .on("end", updateChart) 

    console.log(d3.extent([ [0,0], [width+2*margin,height + 2*margin_y]]) )

    var line = d3.line()
        .x(function(d) { return xScale(d.Episode)  }) 
        .y(function(d) { return yScale(d.Reward); }) 
        .curve(d3.curveMonotoneX)
    
    chart_svg.append("path")
        .attr("class", "line1") 
        .attr("fill", "none")
        .attr("stroke", "orange")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("d", line(data))

    chart_svg.append("path")
        .attr("class", "line2") 
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("d", line(data1))

    chart_svg
    .append("g")
        .attr("class", "brush")
        .call(brush);
        
    /* Add Axis into SVG */
    xAxis = d3.axisBottom(xScale).ticks(12);
    yAxis = d3.axisLeft(yScale).ticks(12);

    chart_svg.append("g")
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

        // What are the selected boundaries?
        extent = d3.event.selection
    
        // If no selection, back to initial coordinate. Otherwise, update X axis domain
        if(!extent){
            // if (!idleTimeout) return idleTimeout = setTimeout(idled, 350); // This allows to wait a little bit
            // xScale.domain([ 4,8])
        }else{
            xScale.domain([ xScale.invert(extent[0]), xScale.invert(extent[1]) ])
            console.log(xScale.invert(extent[0]), xScale.invert(extent[1]))
            chart_svg.select(".brush").call(brush.move, null) // This remove the grey brush area as soon as the selection has been done
        }
    

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

