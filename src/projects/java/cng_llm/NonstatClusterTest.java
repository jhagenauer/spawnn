package cng_llm;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class NonstatClusterTest {

	private static Logger log = Logger.getLogger(NonstatClusterTest.class);
	public static void main(String[] args) {
		Random r = new Random();
		DecimalFormat df = new DecimalFormat("00");

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("output/cluster.shp"),true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : samples)
			desired.add(new double[] { d[3] });

		final int[] fa = new int[] { 2 };
		final int[] ga = new int[] { 0, 1 };
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, 3); // should not be necessary
		
		Map<Integer,Set<double[]>> ref = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int c = (int)d[4];
			if( !ref.containsKey(c) )
				ref.put(c, new HashSet<double[]>());
			ref.get(c).add(d);
		}
		
		final int T_MAX = 20000;
		
		// ------------------------------------------------------------------------

		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		for (int run = 0; run < 1; run++) {
			
			// CNG
			for (int l = 1; l <= 25; l++) {

				List<double[]> neurons = new ArrayList<double[]>();
				for (int i = 0; i < 25; i++) {
					double[] d = samples.get(r.nextInt(samples.size()));
					neurons.add(Arrays.copyOf(d, d.length));
				}

				ErrorSorter errorSorter = new ErrorSorter(samples, desired);
				DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
				Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, l);
				
				DecayFunction nbRate = new PowerDecay(neurons.size()/2, 1);
				DecayFunction lrRate = new PowerDecay(0.5, 0.005);
				LLMNG ng = new LLMNG(neurons, 
						nbRate, lrRate, 
						nbRate, lrRate, 
						sorter, fa, 1, false );
				errorSorter.setLLMNG(ng);

				for (int t = 0; t < T_MAX; t++) {
					int j = r.nextInt(samples.size());
					ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
				}
				Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);
				
				List<double[]> response = new ArrayList<double[]>();
				for (double[] x : samples)
					response.add(ng.present(x));
				log.debug(l+", RMSE: " + Meuse.getRMSE(response, desired) + ", R2: " + Math.pow(Meuse.getPearson(response, desired), 2)+", NMI: "+ClusterValidation.getNormalizedMutualInformation(mapping.values(), ref.values() ));
			}
		}
	}
	
	public static void geoDrawNG(String fn, Map<double[],Double> neurons, Collection<Connection> conections, int[] ga, List<double[]> samples ) {
		int xScale = 1000;
		int yScale = 800;
		
		BufferedImage bufImg = new BufferedImage(xScale, yScale, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.fillRect(0, 0, xScale, yScale);

		// for scaling
		double maxX = Double.MIN_VALUE;
		double minX = Double.MAX_VALUE;
		double maxY = Double.MIN_VALUE;
		double minY = Double.MAX_VALUE;

		for (double[] n : samples ) {
			double x = n[ga[0]];
			double y = n[ga[1]];
			
			if (x > maxX)
				maxX = x;
			if (y > maxY)
				maxY = y;
			if (x < minX)
				minX = x;
			if (y < minY)
				minY = y;
		}
		
		Map<double[],Double> values = new HashMap<double[],Double>();
		for( double[] d : samples )
			values.put(d,d[5]);
		Map<double[],Color> col = ColorBrewerUtil.valuesToColors(values, ColorMode.Set2);
					
		for( double[] n : samples ) {
			g2.setColor(col.get(n));
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			g2.fillOval( x1 - 3, y1 - 3, 6, 6	);
		}

		if( conections != null )
		for( Connection c : conections ) {
			g2.setColor(Color.BLACK);
			double[] a = c.getA();
			double[] b = c.getB();
			int x1 = (int)(xScale * (a[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (a[ga[1]] - minY)/(maxY-minY));
			int x2 = (int)(xScale * (b[ga[0]] - minX)/(maxX-minX));
			int y2 = (int)(yScale * (b[ga[1]] - minY)/(maxY-minY));
			g2.drawLine(x1,y1,x2,y2);
		}

		Map<double[],Color> cMap = ColorBrewerUtil.valuesToColors(neurons, ColorMode.Blues);
		for( double[] n : neurons.keySet() ) {
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			
			g2.setColor(Color.BLACK);
			g2.fillOval( x1 - 8, y1 - 8, 16, 16	);
			
			g2.setColor(cMap.get(n));
			g2.fillOval( x1 - 7, y1 - 7, 14, 14	);
			
			/*g2.setColor(Color.BLACK);
			g2.drawString(""+n.hashCode(),x1,y1);*/
		}
		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", new FileOutputStream(fn));
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
}
