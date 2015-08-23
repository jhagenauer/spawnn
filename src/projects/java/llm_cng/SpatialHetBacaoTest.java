package llm_cng;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
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

public class SpatialHetBacaoTest {

	private static Logger log = Logger.getLogger(SpatialHetBacaoTest.class);

	public static void main(String[] args) {
		Random r = new Random();
		DecimalFormat df = new DecimalFormat("00");

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("/home/julian/git/spawnn/output/bacao.csv"), new int[] { 0, 1 }, new int[] {}, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : samples)
			desired.add(new double[] { d[4] });

		final int[] fa = new int[] { 2, 3 };
		final int[] ga = new int[] { 0, 1 };
		final int T_MAX = 50000;
		
		// ------------------------------------------------------------------------

		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, 4); // should not be necessary
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		for (int run = 0; run < 1; run++) {
			
			// CNG
			for (int l = 3; l <= 3; l++) {

				List<double[]> neurons = new ArrayList<double[]>();
				for (int i = 0; i < 60; i++) {
					double[] d = samples.get(r.nextInt(samples.size()));
					//neurons.add(Arrays.copyOf(d, d.length));
					neurons.add( new double[d.length]);
				}

				// With just errorSorter, we could not distinguish between space. 
				// With fDistsorter instead of errorSorter, we could not distinguish between different relationships
				
				// errorSorter: neurons with low error are mapped more frequently.
				// combined sorter takes spatial distribution and error into account.
				// spatially distribute prototypes, than learn lms is not good, because
				// spatial receptive fields might be homogenous regarding error/consist of differently related 
				// observations
				ErrorSorter errorSorter = new ErrorSorter(samples, desired);
				DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
				Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, l);
				
				DecayFunction nbRate = new PowerDecay(neurons.size()/3, 1);
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
				log.debug(l+", RMSE: " + Meuse.getRMSE(response, desired) + ", R2: " + Math.pow(Meuse.getPearson(response, desired), 2));
				
				// CHL 
				Map<Connection, Integer> conns = new HashMap<Connection, Integer>();
				for (double[] x : samples) {
					sorter.sort(x, ng.getNeurons());
					List<double[]> bmuList = ng.getNeurons();

					Connection c = new Connection(bmuList.get(0), bmuList.get(1));
					if (!conns.containsKey(c))
						conns.put(c, 1);
					else
						conns.put(c, conns.get(c) + 1);
				}
				
				// output/intercept ----------------------------------------------------
				
				Map<double[],double[]> geoOut = new HashMap<double[],double[]>();
				for( double[] n : ng.getNeurons() ) {
					double[] cg = new double[ga.length+1];
					cg[0] = n[ga[0]];
					cg[1] = n[ga[1]];
					cg[2] = ng.output.get(n)[0];
					geoOut.put(n, cg);
				}
				
				List<Connection> outConns = new ArrayList<Connection>();
				for( Connection c : conns.keySet() )
					outConns.add( new Connection( geoOut.get(c.getA()), geoOut.get(c.getB() )) );
								
				// components
				{
					Map<double[],Double> vMap = new HashMap<double[],Double>();
					for( double[] d : ng.getNeurons() ) {
						double[] out = geoOut.get(d);
						vMap.put(out, out[2]);
					}
					geoDrawNG("output/ng_output_"+df.format(l)+"_"+df.format(run)+".png", vMap, outConns, ga, samples);
				}
				
				// coefs ---------------------------------------------------------------
				
				Map<double[],double[]> geoCoefs = new HashMap<double[],double[]>();
				for( double[] n : ng.getNeurons() ) {
					double[] cg = new double[2+fa.length];
					cg[0] = n[ga[0]];
					cg[1] = n[ga[1]];
					for( int i = 0; i < fa.length; i++ )
						cg[2+i] = ng.matrix.get(n)[0][i];
					geoCoefs.put(n, cg);
				}
				
				List<Connection> coefConns = new ArrayList<Connection>();
				for( Connection c : conns.keySet() )
					coefConns.add( new Connection( geoCoefs.get(c.getA()), geoCoefs.get(c.getB() )) );
				
				{ 	// d-matrix
					Map<double[],Double> vMap = new HashMap<double[],Double>();
					for( double[] d : ng.getNeurons() ) {
						DescriptiveStatistics ds = new DescriptiveStatistics();
						double[] coef = geoCoefs.get(d);
						for( double[] nb : Connection.getNeighbors(coefConns, coef, 1))
							ds.addValue( fDist.dist(coef, nb));
						vMap.put( coef, ds.getMean() );
					}
					geoDrawNG("output/ng_coef_dmatrix_"+df.format(l)+"_"+df.format(run)+".png", vMap, coefConns, ga, samples);
				}
				
				// components
				for( int i : fa ) {
					Map<double[],Double> vMap = new HashMap<double[],Double>();
					for( double[] d : ng.getNeurons() ) {
						double[] coef = geoCoefs.get(d);
						vMap.put(coef, coef[i]);
					}
					geoDrawNG("output/ng_coef_v"+i+"_"+df.format(l)+"_"+df.format(run)+".png", vMap, coefConns, ga, samples);
				}
												
				// graph clustering
				Map<double[], Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
				for( Connection c : coefConns ) {
					double[] a = c.getA();
					double[] b = c.getB();
					
					if( !cm.containsKey(a) )
						cm.put(a, new HashSet<double[]>());
					cm.get(a).add(b);
					if( !cm.containsKey(b) )
						cm.put(b, new HashSet<double[]>());
					cm.get(b).add(a);
				}
				Map<Set<double[]>,TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, Clustering.HierarchicalClusteringType.ward);
				List<Set<double[]>> coefCluster = Clustering.cutTree(tree, 3);
				
				List<Set<double[]>> cluster = new ArrayList<Set<double[]>>();
				for( Set<double[]> s : coefCluster ) {
					Set<double[]> c = new HashSet<double[]>();
					for( double[] n : mapping.keySet() ) 
						if( s.contains( geoCoefs.get(n) ) )
							c.addAll(mapping.get(n));
					cluster.add(c);
				}
				
				Map<Integer,Set<double[]>> ref = new HashMap<Integer,Set<double[]>>();
				for( double[] d : samples ) {
					int c = (int)d[5];
					if( !ref.containsKey(c) )
						ref.put(c, new HashSet<double[]>());
					ref.get(c).add(d);
				}
				log.debug("NMI: "+ClusterValidation.getNormalizedMutualInformation(cluster, ref.values() ));
				
				DataUtils.writeCSV("output/cng_coefs_"+df.format(l)+"_"+df.format(run)+".csv", geoCoefs.values(), new String[]{"lat","lon","x1","x2"});
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
