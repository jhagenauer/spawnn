package houseprice.optimSubMarkets;

import java.io.File;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.precision.EnhancedPrecisionOp;
import com.vividsolutions.jts.simplify.DouglasPeuckerSimplifier;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

public class SubmarketsByClusterLM_backup {
	private static Logger log = Logger.getLogger(SubmarketsByClusterLM_backup.class);

	enum method { none, xy, kmedoids, kmeans, cng, kmeans_nomerge };	
	
	public static <GridFeatureBuilder> void main(String[] args) {
		final GeometryFactory gf = new GeometryFactory();
		
		final int[] ga = new int[] { 0, 1 };
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), ga, new int[] {}, true);
		try {
			String crs = "PROJCS[\"unnamed\",GEOGCS[\"Bessel 1841\",DATUM[\"unknown\",SPHEROID[\"bessel\",6377397.155,299.1528128]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"standard_parallel_1\",46],PARAMETER[\"standard_parallel_2\",49],PARAMETER[\"latitude_of_origin\",48],PARAMETER[\"central_meridian\",13.3333333333333],PARAMETER[\"false_easting\",400000],PARAMETER[\"false_northing\",400000],UNIT[\"Meter\",1]]";
			sdf.crs = CRS.parseWKT(crs);
		} catch (NoSuchAuthorityCodeException e1) {
			e1.printStackTrace();
		} catch (FactoryException e1) {
			e1.printStackTrace();
		}
		
		/*SpatialDataFrame bez = DataUtils.readSpatialDataFrameFromShapefile(new File("data/marco/dat4/bez_2008.shp"), true);
		Geometry ab = null;
		for( Geometry g : bez.geoms ) {
			if( ab == null )
				ab = g;
			else
				ab = ab.union(g);
		}
		final Geometry aust = DouglasPeuckerSimplifier.simplify(ab, 200);*/
		
		Geometry ab = null;
		try {
			SpatialDataFrame bez = DataUtils.readSpatialDataFrameFromShapefile(new File("data/nuts_austria/STATISTIK_AUSTRIA_NUTS1_20160101.shp"), true);
			ab = DouglasPeuckerSimplifier.simplify( JTS.transform(bez.geoms.get(0), CRS.findMathTransform(bez.crs, sdf.crs,true)),50);
		} catch (MismatchedDimensionException e1) {
			e1.printStackTrace();
		} catch (TransformException e1) {
			e1.printStackTrace();
		} catch (FactoryException e1) {
			e1.printStackTrace();
		}
		final Geometry aust = ab;
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		for (double[] d : sdf.samples) {
			if( !aust.intersects(gf.createPoint(new Coordinate(d[ga[0]], d[ga[1]]))) )
				continue;
			
			double[] nd = Arrays.copyOf(d, d.length - 1);		
			samples.add(nd);
			desired.add(new double[] { d[d.length - 1] });		
		}
		log.debug(samples.size()+"/"+sdf.samples.size());

		final int[] fa = new int[samples.get(0).length - 2]; // omit geo-vars
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		
		//DataUtils.transform(samples, fa, transform.zScore);
		//DataUtils.transform(desired, new int[]{0}, transform.zScore);

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
						
		List<Entry<List<Integer>,List<Integer>>> permutations = new ArrayList<Entry<List<Integer>,List<Integer>>>();
		for( int repeat = 0; repeat < 1; repeat++ ) {
			
			// full
			/*List<Integer> l = new ArrayList<Integer>();
			for( int i = 0; i < samples.size(); i++ )
				l.add(i);
			Collections.shuffle(l);
			List<Integer> train = new ArrayList<Integer>(l);
			List<Integer> val = new ArrayList<Integer>(l);
			permutations.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train,val) );*/
			
			// cv
			int numFolds = 10; 
			List<Integer> l = new ArrayList<Integer>();
			for( int i = 0; i < samples.size(); i++ )
				l.add(i);
			Collections.shuffle(l);
			int foldSize = samples.size()/numFolds;
			for (int fold = 0; fold < numFolds; fold++) {				
				List<Integer> val = new ArrayList<Integer>(l.subList(fold*foldSize, (fold+1)*foldSize));
				List<Integer> train = new ArrayList<Integer>(l);
				train.removeAll(val);
				permutations.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train,val) );
			}
		}
		log.debug("perms: "+permutations.size());
				
		for( int numMedoids : new int[]{ 10} )
		for( final method m : new method[]{ /*method.none, method.xy, */method.kmeans_nomerge/*, method.kmeans /*, method.cng*/ } ) {
		
		ExecutorService es = Executors.newFixedThreadPool(3);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
						
		long time = System.currentTimeMillis();
		for( final Entry<List<Integer>,List<Integer>> perm : permutations ) {
			final int nm = numMedoids;

			final List<double[]> samplesTrain = new ArrayList<double[]>();
			final List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int i : perm.getKey() ) {
				samplesTrain.add(samples.get(i));
				desiredTrain.add(desired.get(i));
			}
			
			final List<double[]> samplesVal = new ArrayList<double[]>();
			final List<double[]> desiredVal = new ArrayList<double[]>();
			for( int i : perm.getValue() ) {
				samplesVal.add(samples.get(i));
				desiredVal.add(desired.get(i));
			}			
			
			futures.add(es.submit(new Callable<double[]>() {
				@Override
				public double[] call() throws Exception {
					if( m == method.none ) {
						List<Set<double[]>> l = new ArrayList<Set<double[]>>();
						Set<double[]> s = new HashSet<double[]>();
						s.addAll(samplesTrain);
						s.addAll(samplesVal);
						l.add(s);
						return new double[]{ getLMwithDummyCost( l, samplesTrain, desiredTrain, samplesVal, desiredVal, fa) };
					} else if( m == method.xy ) {
						List<Set<double[]>> l = new ArrayList<Set<double[]>>();
						Set<double[]> s = new HashSet<double[]>();
						s.addAll(samplesTrain);
						s.addAll(samplesVal);
						l.add(s);
						int[] nfa = new int[ga.length+fa.length];
						for( int i = 0; i < ga.length; i++ )
							nfa[i] = ga[i];
						for( int i = 0; i < fa.length; i++ )
							nfa[i+ga.length] = fa[i];
						return new double[]{ getLMwithDummyCost( l, samplesTrain, desiredTrain, samplesVal, desiredVal, nfa) };
					} else if( m == method.kmeans_nomerge ) { 
						Map<double[], Set<double[]>> c = null;
						double min = Double.MAX_VALUE;
						for( int i = 0; i < 100; i++ ) {
							Map<double[], Set<double[]>> can = Clustering.kMeans(samplesTrain, nm, gDist);
							if( DataUtils.getMeanQuantizationError(can, gDist) < min ) {
								min = DataUtils.getMeanQuantizationError(can, gDist);
								c = can;
								i = 0;
							}
						}
						
						// add val samples 
						for (double[] d : samplesVal) {
							double[] minCenter = null;
							for (double[] center : c.keySet())
								if (minCenter == null || gDist.dist(center, d) < gDist.dist(minCenter, d))
									minCenter = center;
							c.get(minCenter).add(d);
						}
						
						double cost = getLMwithDummyCost( new ArrayList<Set<double[]>>(c.values()), samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
						return new double[]{cost};
					} else {
						// split train
						Random r = new Random();
						List<double[]> subSamplesTrain = new ArrayList<double[]>(samplesTrain);
						List<double[]> subDesiredTrain = new ArrayList<double[]>(desiredTrain);
						List<double[]> subSamplesVal = new ArrayList<double[]>();
						List<double[]> subDesiredVal = new ArrayList<double[]>();
						while( subSamplesVal.size() < 0.3*subSamplesTrain.size() ) {
							int idx = r.nextInt(subSamplesTrain.size());
							subSamplesVal.add( subSamplesTrain.remove(idx));
							subDesiredVal.add( subDesiredTrain.remove(idx));
						}
																		
						Map<double[], Set<double[]>> c = null;
						if( m == method.kmedoids ) {
							c = Clustering.kMedoidsPAM(subSamplesTrain, nm, gDist);
						} else if( m == method.kmeans ) {
							//c = Clustering.kMeans(subSamplesTrain, nm, gDist);
							
							double min = Double.MAX_VALUE;
							for( int i = 0; i < 100; i++ ) {
								Map<double[], Set<double[]>> can = Clustering.kMeans(subSamplesTrain, nm, gDist);
								if( DataUtils.getMeanQuantizationError(can, gDist) < min ) {
									min = DataUtils.getMeanQuantizationError(can, gDist);
									c = can;
									i = 0;
								}
							}
							
						} else if( m == method.cng ) {
							Sorter<double[]> sorter = new KangasSorter<double[]>(gDist, fDist, 1);
							DecayFunction nbRange = new PowerDecay( nm/4, 0.001 );
							DecayFunction adaptRate = new PowerDecay( 1.0, 0.001 );
							
							List<double[]> neurons = new ArrayList<double[]>();
							while( neurons.size() != nm ) {
								double[] d = subSamplesTrain.get(r.nextInt(subSamplesTrain.size()));
								neurons.add( Arrays.copyOf(d, d.length) );
							}
							NG ng = new NG(neurons, nbRange, adaptRate, sorter);
							int T_MAX = 100000;
							for (int t = 0; t < T_MAX; t++) 
								ng.train((double) t / T_MAX, subSamplesTrain.get(r.nextInt(subSamplesTrain.size())));
							
							c = NGUtils.getBmuMapping(subSamplesTrain, neurons, sorter);
						} 
						
						// merge
						List<double[]> centers = new ArrayList<double[]>(c.keySet());
						List<Coordinate> coords = new ArrayList<Coordinate>();
						for (double[] d : centers)
							coords.add(new Coordinate(d[ga[0]], d[ga[1]]));
						VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
						vdb.setClipEnvelope(aust.getEnvelopeInternal());
						//vdb.setTolerance(0.001);
						vdb.setSites(coords);
						GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

						List<Geometry> voroGeoms = new ArrayList<Geometry>();
						for (int i = 0; i < coords.size(); i++) {
							Geometry p = gf.createPoint(coords.get(i));
							for (int j = 0; j < coll.getNumGeometries(); j++) 
								if (p.intersects(coll.getGeometryN(j))) {
									voroGeoms.add(EnhancedPrecisionOp.intersection(coll.getGeometryN(j), aust));
									break;
								}
						}
												
						// adding val samples to obtain cluster-connect-map, changes clusters!
						for (double[] d : subSamplesVal) {
							double[] minCenter = null;
							for (double[] center : c.keySet())
								if (minCenter == null || gDist.dist(center, d) < gDist.dist(minCenter, d))
									minCenter = center;
							c.get(minCenter).add(d);
						}

						// build cm map based on voro
						Map<Set<double[]>, Set<Set<double[]>>> cm = new HashMap<Set<double[]>, Set<Set<double[]>>>();
						for (int i = 0; i < centers.size(); i++) {
							Set<Set<double[]>> s = new HashSet<Set<double[]>>();
							for (int j = 0; j < centers.size(); j++) 
								if (i != j && voroGeoms.get(i).intersects(voroGeoms.get(j))) 
									s.add(c.get(centers.get(j)));
							cm.put(c.get(centers.get(i)), s);
						}
												
						Map<Set<double[]>,TreeNode> tree = getHierarchicalClustering(c.values(), cm, subSamplesTrain, subDesiredTrain, subSamplesVal, subDesiredVal, fa);
																								
						// cut tree
						Comparator<TreeNode> comp = new Comparator<TreeNode>() { // oldest first
							@Override
							public int compare(TreeNode o1, TreeNode o2) {
								return -Integer.compare(o1.age, o2.age);
							}
						};						
	
						// get sequential costs based on samplesVa for each clustering/cut!
						PriorityQueue<TreeNode> leafLayer = new PriorityQueue<TreeNode>(1, comp);
						leafLayer.add(  Collections.min(tree.values(), comp) );
						double bestCost = Double.MAX_VALUE;
						List<Set<double[]>> bestMergedC = null;
						while( true ) {
							TreeNode tn = leafLayer.peek();
							
							if( tn.cost < bestCost ) {
								bestCost = tn.cost;
								
								bestMergedC = new ArrayList<Set<double[]>>();
								for( Entry<Set<double[]>,TreeNode> e : tree.entrySet() )
									if( leafLayer.contains(e.getValue()))
										bestMergedC.add(e.getKey());
							}
											
							boolean b = true;
							for( TreeNode n : leafLayer )
								if( n.age != 0 )
									b = false;
							if( b )
								break;
															
							leafLayer.remove();
														
							if( tn.children != null )
								for( TreeNode child : tn.children )
									if( child != null )
										leafLayer.add(child);
						}	
						
						// get final model, add samples to mergedC, based on distance to centers
						for( double[] d : samplesVal ) {

							// find closest center
							double[] minCenter = null;
							for( double[] center : c.keySet() )
								if( minCenter == null || gDist.dist(d, center) < gDist.dist(d, minCenter ) )
									minCenter = center;

							// find merged cluster of center and add d
							Set<double[]> clust = null;
							for( Set<double[]> s : bestMergedC ) {
								Set<double[]> is = new HashSet<double[]>(c.get(minCenter));
								is.retainAll(s);
								if( !is.isEmpty() ) {
									clust = s;
									break;
								}
							}
							clust.add(d);
						}

						double finalCost = getLMwithDummyCost(bestMergedC, samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
												
						// draw best of models
						List<Set<double[]>> cCluster = new ArrayList<Set<double[]>>();
						for( Set<double[]> s : bestMergedC ) {
							Set<double[]> cs = new HashSet<double[]>();
							for( double[] d : s ) {
								double[] minCenter = null;
								for( double[] center : c.keySet() ) {
									if( minCenter == null || gDist.dist(d, center) < gDist.dist(d, minCenter ) )
										minCenter = center;
								}
								cs.add(minCenter);
							}
							cCluster.add(cs);
						}
						Drawer.geoDrawCluster(cCluster, centers, voroGeoms, "output/"+this.hashCode()+"_"+nm+"_"+cCluster.size()+"_"+finalCost+".png", true);
												
						return new double[]{finalCost};
					}
				}
			}));
		}
		es.shutdown();

 		double[] mean = null;
		for (Future<double[]> ff : futures) {
			try {
				double[] ee = ff.get();
				if (mean == null) 
					mean = ee;
				else
					for( int i = 0; i < mean.length; i++ )
						mean[i] += ee[i];
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			}
		}	
		double took = (System.currentTimeMillis()-time)/(60000.0);	
		double minCost = Double.MAX_VALUE;
		for( int i = 0; i < mean.length; i++ )
			if( mean[i] < minCost ) {
				minCost = mean[i];
			}
		log.debug("centers: "+numMedoids+", method: "+m+", cost: "+(minCost/futures.size())+", took: "+took);
		}
	}
	
	static boolean selRandom = false;
	static Map<Set<double[]>,TreeNode>  getHierarchicalClustering( Collection<Set<double[]>> cluster, Map<Set<double[]>, Set<Set<double[]>>> cm, List<double[]> samplesTrain, List<double[]> desiredTrain, List<double[]> samplesVal, List<double[]> desiredVal, int[] fa) {		
		class FlatSet<T> extends HashSet<T> {
			private static final long serialVersionUID = -1960947872875758352L;
			public int hashCode = super.hashCode();
			
			@Override 
			public boolean add( T t ) {
				hashCode += t.hashCode();
				return super.add(t);
			}
			
			@Override
			public boolean addAll( Collection<? extends T> c ) {
				hashCode += c.hashCode();
				return super.addAll(c);
			}
			
			@Override
			public int hashCode() {
				return hashCode;
			}
		}
		
		// init
		List<Set<double[]>> leafLayer = new ArrayList<Set<double[]>>();
		for (Set<double[]> d : cluster ) {
			Set<double[]> l = new FlatSet<double[]>();
			l.addAll(d);
			leafLayer.add(l);
		}
					
		// init connected map
		Map<Set<double[]>, Set<Set<double[]>>> connected = new HashMap<Set<double[]>, Set<Set<double[]>>>();
		for (Set<double[]> a : cluster) {
			
			Set<Set<double[]>> lfNbs = new HashSet<Set<double[]>>();
			for (Set<double[]> nb : cm.get(a) ) 
				for( Set<double[]> lf : leafLayer )
					if( lf.containsAll(nb) )
						lfNbs.add(lf);
						
			for( Set<double[]> lf : leafLayer )
				if( lf.containsAll(a) )
					connected.put(lf, lfNbs);
		}

		int age = 0;	
		Map<Set<double[]>,TreeNode> tree = new HashMap<Set<double[]>,TreeNode>();
		double curCost = getLMwithDummyCost(leafLayer, samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
		for( Set<double[]> s : leafLayer ) {
			TreeNode cn = new TreeNode();
			cn.age = age;
			cn.cost = curCost;
			tree.put(s,cn);
		}
		
		while (leafLayer.size() > 1 ) {
			final Map<int[],Double> costs = new HashMap<int[],Double>();
			for (int i = 0; i < leafLayer.size() - 1; i++) {
				Set<double[]> l1 = leafLayer.get(i);

				for (int j = i + 1; j < leafLayer.size(); j++) {
					Set<double[]> l2 = leafLayer.get(j);
					
					if( !connected.containsKey(l1) || !connected.get(l1).contains(l2) ) // disjoint
						continue;
									
					Set<double[]> union = new FlatSet<double[]>();
					union.addAll(l1);
					union.addAll(l2);
															
					leafLayer.remove(l1);
					leafLayer.remove(l2);
					leafLayer.add(union);
															
					double increase = getLMwithDummyCost(leafLayer, samplesTrain, desiredTrain, samplesVal, desiredVal, fa) - curCost;
																
					leafLayer.remove(union);					
					leafLayer.add(i,l1);
					leafLayer.add(j,l2);
					
					costs.put( new int[]{i,j}, increase);
				}
			}
			

			Set<double[]> c1 = null, c2 = null;
			double selIncrease = Double.MAX_VALUE;
			
			// chose min
			for( Entry<int[],Double> e : costs.entrySet() )
				if( e.getValue() < selIncrease ) {
					selIncrease = e.getValue();
					c1 = leafLayer.get(e.getKey()[0]);
					c2 = leafLayer.get(e.getKey()[1]);
				}
			
			// chose randomly
			if( selRandom ) {
				double max = Collections.max(costs.values());
				double sum = 0; 
				for( double d : costs.values() )
					sum += Math.abs(d-max);
				double r = new Random().nextDouble()*sum;
				double cur = 0;
				for( Entry<int[],Double> e : costs.entrySet() ) {
					double cv = Math.abs(e.getValue()-max);
					if( cur <= r && r < cur + cv ) {
						selIncrease = e.getValue();
						c1 = leafLayer.get(e.getKey()[0]);
						c2 = leafLayer.get(e.getKey()[1]);
						break;
					}
					sum += cv;
				}
			}
																			
			// merge
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(c1);
			union.addAll(c2);	
						
			leafLayer.add(union);
			leafLayer.remove(c1);
			leafLayer.remove(c2);
									
			// update connected map
			// 1. merge values of c1 and c2 and put union
			Set<Set<double[]>> ns = connected.remove(c1);
			ns.addAll( connected.remove(c2) );
			connected.put(union, ns);
			
			// 2. replace all values c1,c2 by union
			for( Set<double[]> a : connected.keySet() ) {
				Set<Set<double[]>> s = connected.get(a);
				if( s.contains(c1) || s.contains(c2)) {
					s.remove(c1);
					s.remove(c2);
					s.add(union);
				}
			}
			curCost += selIncrease;
			
			TreeNode cn = new TreeNode();
			cn.cost = curCost;
			cn.children = Arrays.asList( new TreeNode[]{ tree.get(c1), tree.get(c2) } );
			cn.age = ++age;
			tree.put(union, cn);
		}
		return tree;
	}
			
	public static double getLMwithDummyCost( List<Set<double[]>> cluster, List<double[]> samplesTrain, List<double[]> desiredTrain , List<double[]> samplesVal, List<double[]> desiredVal, int[] fa) {
		// check that all samples are assigned to a cluster
		Set<double[]> all = new HashSet<double[]>();
		int sumSize = 0;
		for( Set<double[]> s : cluster ) {
			sumSize+=s.size();
			all.addAll(s);
		}
		if( sumSize != all.size() )
			throw new RuntimeException("Cluster overlap!");
		
		if( !all.containsAll(samplesTrain) || !all.containsAll(samplesVal) )
			throw new RuntimeException("Some samples not assigend to cluster!");
				
		double[] y = new double[desiredTrain.size()];
		for (int i = 0; i < desiredTrain.size(); i++)
			y[i] = desiredTrain.get(i)[0];

		double[][] x = new double[samplesTrain.size()][];
		for (int i = 0; i < samplesTrain.size(); i++) {
			double[] d = samplesTrain.get(i);
			x[i] = getStripped(d, fa);
			
			// add 1-column
			/*double[] nxi = new double[x[i].length+1];
			nxi[0] = 1;
			for( int j = 0; j < x[i].length; j++ )
				nxi[j+1] = x[i][j];
			x[i] = nxi;*/
							
			int length = x[i].length;
			x[i] = Arrays.copyOf(x[i], length + cluster.size() - 1);
			for( int idx = 0; idx < cluster.size()-1; idx++ ) {
				if( cluster.get(idx).contains(d) ) {
					x[i][length + idx] = 1;
					break;
				}
			}				
		}
					
		// training
		double[] beta = null;
		try {
			OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
			ols.setNoIntercept(false);
			ols.newSampleData(y, x);
			beta = ols.estimateRegressionParameters();
			
			/*DoubleMatrix A = new DoubleMatrix(x);
			DoubleMatrix B = new DoubleMatrix(y);
			DoubleMatrix X = Solve.solveLeastSquares(A, B);
			beta = X.toArray();*/
						
		} catch( Exception e ) {
			List<Integer> l = new ArrayList<Integer>();
			for( Set<double[]> s : cluster )
				l.add(s.size());
			Collections.sort(l);
			System.out.println(l);
			e.printStackTrace();
			System.exit(1);
		}
		
		// testing/AIC 
		/*List<double[]> responseTrain = new ArrayList<double[]>();
		for( int i = 0; i < samplesTrain.size(); i++ ) {
			double[] d = samplesTrain.get(i);
			double[] xi = getStripped(d, fa);
			
			int length = xi.length;
			xi = Arrays.copyOf(xi, length + cluster.size() - 1);
			for( int idx = 0; idx < cluster.size(); idx++ ) {			
				if( cluster.get(idx).contains(d) ) {
					if( idx < cluster.size()-1 )
						xi[length+idx] = 1;
					break;
				}
			}
			double p = beta[0]; // intercept at beta[0]
			for (int j = 1; j < beta.length; j++)
				p += beta[j] * xi[j - 1];

			responseTrain.add(new double[] { p });
		}
		return Meuse.getAIC( Meuse.getMSE(responseTrain, desiredTrain), x[0].length+1, x.length);*/
		
		// testing/RMSE
		List<double[]> responseVal = new ArrayList<double[]>();
		for (int i = 0; i < samplesVal.size(); i++) {
			double[] d = samplesVal.get(i);
			double[] xi = getStripped(d, fa);
			
			int length = xi.length;
			xi = Arrays.copyOf(xi, length + cluster.size() - 1);
			for( int idx = 0; idx < cluster.size(); idx++ ) {			
				if( cluster.get(idx).contains(d) ) {
					if( idx < cluster.size()-1 )
						xi[length+idx] = 1;
					break;
				}
			}
			
			double p = beta[0]; // intercept at beta[0]
			for (int j = 1; j < beta.length; j++)
				p += beta[j] * xi[j - 1];

			responseVal.add(new double[] { p });
		}
		return Meuse.getRMSE(responseVal, desiredVal);
	}
	
	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
