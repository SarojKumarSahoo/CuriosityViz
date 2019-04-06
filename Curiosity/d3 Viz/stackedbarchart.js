d3.csv('episode_data.csv', function(p_data)  {
    data = p_data;
    plot_it(data);
})



function plot_it(data){

    keys = d3.keys(data[0])
    console.log(d3.stack().keys(keys.slice(2,5))(data))
    eps = [];
    eps_r = []
    eps_irl = []
    for (i=0; i<data.length; i++)
    {
        eps.push(i+1)
        eps_r.push(data[i]['Episode reward'])
        eps_irl.push(data[i]['Episode reward irl'])

    }

    var width = 1670;
    var height = 400;
    var padding = 10;
    var actual_width = width - 2*padding;

    var xScale = d3.scaleBand().domain(eps).range([padding,actual_width]).paddingInner(0.01);
    var yScale = d3.scaleLinear().domain([0,520]).range([200, 0]);
    var z = d3.scaleOrdinal(d3.schemeCategory20c);

    z.domain(keys.slice(2,5))
    var svg = d3.select('#stackedBarChart').append('svg').attr('width', width).attr('height', height);

    svg.append("g")
    .selectAll("g")
    .data(d3.stack().keys(keys.slice(2,5))(data))
    .enter().append("g")
      .attr("fill", function(d) { 
        return z(d.key); })
    .selectAll("rect")
    .data(function(d) { return d; })
    .enter().append("rect")
      .attr("x", function(d) { return xScale(d.data['Episode number']); })
      .attr("y", function(d) { return yScale(d[1]); })
      .attr("height", function(d) { return yScale(d[0]) - yScale(d[1]); })
      .attr("width", xScale.bandwidth())
      .on("click", function(d){
            get_graph(d.data['Episode number'])
            get_graph_1(d.data['Episode number'])

            get_graph_2(d.data['Episode number'])
        });

      
}




