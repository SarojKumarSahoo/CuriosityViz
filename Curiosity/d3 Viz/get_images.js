var svg = d3.select("#image1")
    .append("svg")
    .attr("width", 200)
    .attr("height", 100)
    .style("border", "1px solid black");

var text = svg.selectAll("text")
    .data([0])
    .enter()
    .append("text")
    .text("Testing")
    .attr("x", "40")
    .attr("y", "60");

var imgs = svg.selectAll("image").data([0]);
imgs.enter()
    .append("svg:img")
    .attr("xlink:href", "‎⁨file:///Users⁩/saroj⁩/Viz⁩/project⁩/frames⁩/frame_0.png")
    .attr("x", "60")
    .attr("y", "60")
    .attr("width", "20")
    .attr("height", "20");
