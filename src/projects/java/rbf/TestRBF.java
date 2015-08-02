package rbf;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.AdaptIncRBF;
import spawnn.rbf.RBF;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;

public class TestRBF {

	private static Logger log = Logger.getLogger(TestRBF.class);

	public static void main(String[] args) {
				
		Random r = new Random();

		List<double[]> samples = new ArrayList<double[]>();
		List<double[]> desired = new ArrayList<double[]>();
		
		while( samples.size() < 1000 ) {
			double x = r.nextDouble()*2-1;
			samples.add( new double[]{x} );
			if( x < 0 ) {
				desired.add( new double[]{-1} );
			} else {
				desired.add( new double[]{+1} );
			}
		}
				
		Dist<double[]> dist = new EuclideanDist();

		/*Map<double[], Double> hidden = new HashMap<double[], Double>();
		Map<double[], Set<double[]>> clustering = Clustering.kMeans(samples, 2, dist);
		double qe = DataUtils.getQuantError(clustering, dist);
		for( int i = 0; i < 100; i++ ) {
			Map<double[], Set<double[]>> tmp = Clustering.kMeans(samples, clustering.size(), dist);
			if( DataUtils.getQuantError(tmp, dist) < qe ) {
				qe = DataUtils.getQuantError(tmp, dist);
				clustering = tmp;
			}
		}
		
		for (double[] c : clustering.keySet()) {
			double d = Double.MAX_VALUE;
			for (double[] n : clustering.keySet())
				if (c != n)
					d = Math.min(d, dist.dist(c, n));
			log.debug(Arrays.toString(c)+","+d);
			hidden.put(c, 0.5*d);
		}
		
		RBF rbf = new RBF(hidden, 1, dist, 0.05);*/
		
		Map<double[], Double> hidden = new HashMap<double[], Double>();
		while (hidden.size() < 2) {
			double[] d = samples.get(r.nextInt(samples.size()));
			hidden.put(Arrays.copyOf(d, d.length), 1.0);
		}
		//RBF rbf = new IncRBF(hidden, 0.05, 0.0005, dist, 100, -1, 0.5, 0.0005, 0.05, 1);
		RBF rbf = new AdaptIncRBF(hidden, 0.05, 0.0005, dist, 100, 0.1, 0.5, 0.0005, 0.05, 1);

		for (int i = 0; i < 10000; i++) {
			int j = r.nextInt(samples.size());
			rbf.train(samples.get(j), desired.get(j));
		}
		
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples)
			response.add(rbf.present(x));
		
		log.debug("rmse: "+Meuse.getRMSE(response, desired) ); 
		log.debug("r^2: "+Math.pow(Meuse.getPearson(response, desired), 2) ); 
		
		// plot output function
		XYSeriesCollection dataset = new XYSeriesCollection();
		
		{
			XYSeries s = new XYSeries("data");
			for( int i = 0; i < samples.size(); i++ ) 
				s.add(samples.get(i)[0],desired.get(i)[0]);
			dataset.addSeries(s);
		}
		
		int i = 0;
		for( double[] n : rbf.getNeurons().keySet() ) {
			XYSeries s = new XYSeries("neuron "+(i++));	
			for( double d = -1; d <= 1; d+=0.01 )
				s.add(d, Math.exp( -0.5 * Math.pow(dist.dist( new double[]{d}, n) / rbf.getNeurons().get(n), 2) ) );
			dataset.addSeries(s);
		}
		
		XYSeries s = new XYSeries("rbf");	
		for( double d = -1; d <= 1; d+=0.01 )
			s.add(d, rbf.present( new double[]{d} )[0] );
		dataset.addSeries(s);
				
		JFreeChart lineplot = ChartFactory.createXYLineChart("","x","y",dataset,PlotOrientation.VERTICAL,true,true,false);
		try {
			ChartUtilities.saveChartAsPNG(new File("output/rbf_output.png"), lineplot, 1024, 768);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}
