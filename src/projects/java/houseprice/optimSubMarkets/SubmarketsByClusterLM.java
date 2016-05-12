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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;

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

public class SubmarketsByClusterLM {
	private static Logger log = Logger.getLogger(SubmarketsByClusterLM.class);

	enum method { none, xy, nuts3, kmedoids, kmeans, cng, kmeans_nm };	
	enum measure {AIC, RMSE }

	static double globalBest = Double.MAX_VALUE;
		
	static class Validation {
		Validation(measure v, int folds, int repeats) {
			this.v = v; this.folds = folds; this.repeats = repeats; 
			if( v == measure.AIC && folds != 0 ) {
				log.warn("Setting folds to 0");
				this.folds = 0;
			}
		}
		measure v;
		int folds = 0; 
		int repeats = 1;
		public String toString() { return v+","+folds+","+repeats; }
	}
		
	public static <GridFeatureBuilder> void main(String[] args) {
		final GeometryFactory gf = new GeometryFactory();
		
		Geometry ab = null;
		final int[] ga = new int[] { 0, 1 };
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), ga, new int[] {}, true);
		final List<Geometry> nuts3Geoms = new ArrayList<Geometry>();
		try {
			String crs = "PROJCS[\"unnamed\",GEOGCS[\"Bessel 1841\",DATUM[\"unknown\",SPHEROID[\"bessel\",6377397.155,299.1528128]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"standard_parallel_1\",46],PARAMETER[\"standard_parallel_2\",49],PARAMETER[\"latitude_of_origin\",48],PARAMETER[\"central_meridian\",13.3333333333333],PARAMETER[\"false_easting\",400000],PARAMETER[\"false_northing\",400000],UNIT[\"Meter\",1]]";
			sdf.crs = CRS.parseWKT(crs);
			
			SpatialDataFrame nuts1 = DataUtils.readSpatialDataFrameFromShapefile(new File("data/nuts_austria/STATISTIK_AUSTRIA_NUTS1_20160101.shp"), true);
			ab = DouglasPeuckerSimplifier.simplify( JTS.transform(nuts1.geoms.get(0), CRS.findMathTransform(nuts1.crs, sdf.crs,true)),50);
			
			SpatialDataFrame nuts3 = DataUtils.readSpatialDataFrameFromShapefile(new File("data/nuts_austria/STATISTIK_AUSTRIA_NUTS3_20160101.shp"), true);
			for (Geometry g : nuts3.geoms.subList(0, 11))
				nuts3Geoms.add(JTS.transform(g, CRS.findMathTransform(nuts3.crs, sdf.crs, true)));
		} catch (Exception e) {
			e.printStackTrace();
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
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
				
		final Validation outerVal = new Validation(measure.RMSE, 0, 25);
		//final Validation innerVal = new Validation(measure.AIC, 0, 1); 
			

		for( int numMedoids : new int[]{ 15, 20, 25, 30, 35, 40 } ) {
		for( final Validation innerVal : new Validation[]{
				new Validation(measure.AIC, 0, 1),
				//new Validation(measure.RMSE, 10, 4)
				} )
		for( final method m : new method[]{ /*method.none, method.xy, method.nuts3, method.kmeans_nm,*/ method.kmeans /*, method.cng*/ } ) {
					
		ExecutorService es = Executors.newFixedThreadPool(4);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
						
		long time = System.currentTimeMillis();
		for( final Entry<List<Integer>,List<Integer>> cvEntry : getCVList(outerVal.folds, outerVal.repeats, samples.size()) ) {
			final int nm = numMedoids;

			final List<double[]> samplesTrain = new ArrayList<double[]>();
			final List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int i : cvEntry.getKey() ) {
				samplesTrain.add(samples.get(i));
				desiredTrain.add(desired.get(i));
			}
			
			final List<double[]> samplesVal = new ArrayList<double[]>();
			final List<double[]> desiredVal = new ArrayList<double[]>();
			for( int i : cvEntry.getValue() ) {
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
						
						if( outerVal.v == measure.AIC )
							return new double[]{ getAICofLM( l, samplesTrain, desiredTrain, fa) };
						else
							return new double[]{ getRMSEofLM( l, samplesTrain, desiredTrain, samplesVal, desiredVal, fa) };
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
						
						if( outerVal.v == measure.AIC )
							return new double[]{ getAICofLM( l, samplesTrain, desiredTrain, nfa) };
						else
							return new double[]{ getRMSEofLM( l, samplesTrain, desiredTrain, samplesVal, desiredVal, nfa) };
					} else if( m == method.kmeans_nm ) { 
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
						
						if( outerVal.v == measure.AIC )
							return new double[]{ getAICofLM(new ArrayList<Set<double[]>>(c.values()), samplesTrain, desiredTrain, fa) };
						else
							return new double[]{ getRMSEofLM( new ArrayList<Set<double[]>>(c.values()), samplesTrain, desiredTrain, samplesVal, desiredVal, fa) };
					} else if( m == method.nuts3 ) {
						Map<Geometry,Set<double[]>> c = new HashMap<Geometry,Set<double[]>>();
						for( Geometry g : nuts3Geoms ) {
							Set<double[]> s = new HashSet<double[]>();
							for( double[] d : samples )
								if( gf.createPoint( new Coordinate(d[ga[0]], d[ga[1]])).intersects(g) )
									s.add(d);
							c.put(g, s);
						}
						
						if( outerVal.v == measure.AIC )
							return new double[]{ getAICofLM(new ArrayList<Set<double[]>>(c.values()), samplesTrain, desiredTrain, fa) };
						else
							return new double[]{ getRMSEofLM( new ArrayList<Set<double[]>>(c.values()), samplesTrain, desiredTrain, samplesVal, desiredVal, fa) };
					} else { // greedy merge
						Map<double[], Set<double[]>> c = null;
						if( m == method.kmedoids ) {
							c = Clustering.kMedoidsPAM(samplesTrain, nm, gDist);
						} else if( m == method.kmeans ) {
							//c = Clustering.kMeans(subSamplesTrain, nm, gDist);
							
							double min = Double.MAX_VALUE;
							for( int i = 0; i < 100; i++ ) {
								Map<double[], Set<double[]>> can = Clustering.kMeans(samplesTrain, nm, gDist);
								if( DataUtils.getMeanQuantizationError(can, gDist) < min ) {
									min = DataUtils.getMeanQuantizationError(can, gDist);
									c = can;
									i = 0;
								}
							}
							
						} else if( m == method.cng ) {
							Random r = new Random();
							Sorter<double[]> sorter = new KangasSorter<double[]>(gDist, fDist, 1);
							DecayFunction nbRange = new PowerDecay( nm/4, 0.001 );
							DecayFunction adaptRate = new PowerDecay( 1.0, 0.001 );
							
							List<double[]> neurons = new ArrayList<double[]>();
							while( neurons.size() != nm ) {
								double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
								neurons.add( Arrays.copyOf(d, d.length) );
							}
							NG ng = new NG(neurons, nbRange, adaptRate, sorter);
							int T_MAX = 100000;
							for (int t = 0; t < T_MAX; t++) 
								ng.train((double) t / T_MAX, samplesTrain.get(r.nextInt(samplesTrain.size())));
							
							c = NGUtils.getBmuMapping(samplesTrain, neurons, sorter);
						} 
						
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
						for (double[] d : samplesVal) {
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
												
						Map<Set<double[]>,TreeNode> tree = getHierarchicalClustering(c.values(), cm, samplesTrain, desiredTrain, fa, innerVal );
																								
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
												
						double finalCost;
						if( outerVal.v == measure.AIC )
							finalCost = getAICofLM(bestMergedC, samples, desired, fa);
						else
							finalCost = getRMSEofLM(bestMergedC, samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
						
						if( finalCost < globalBest ) {
							synchronized (this) {
								globalBest = finalCost;
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
							}
						}
						
						return new double[]{ finalCost };
					}
				}
			}));
		}
		es.shutdown();

 		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (Future<double[]> ff : futures) {
			try {
				ds.addValue(ff.get()[0]);
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			}
		}	
		double took = (System.currentTimeMillis()-time)/(60000.0);	
		
		log.info(innerVal+", centers: "+numMedoids+", method: "+m+", cost: "+ds.getMean()+" ("+ds.getStandardDeviation()+")"+", took: "+took);
		}
		}
	}
	
	static Map<Set<double[]>,TreeNode>  getHierarchicalClustering( Collection<Set<double[]>> cluster, Map<Set<double[]>, Set<Set<double[]>>> cm, List<double[]> samples, List<double[]> desired, int[] fa, Validation val ) {		
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
		
		double curCost = 0;
		List<Entry<List<Integer>,List<Integer>>> cvList = null;
		if( val.v == SubmarketsByClusterLM.measure.AIC ) {
			curCost = getAICofLM(leafLayer, samples, desired, fa);
		} else {
			cvList = getCVList(val.folds, val.repeats,  samples.size());
			for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList ) {
				List<double[]> samplesTrain = new ArrayList<double[]>();
				List<double[]> desiredTrain = new ArrayList<double[]>();
				for( int k : cvEntry.getKey() ) {
					samplesTrain.add(samples.get(k));
					desiredTrain.add(desired.get(k));
				}
				
				List<double[]> samplesVal = new ArrayList<double[]>();
				List<double[]> desiredVal = new ArrayList<double[]>();
				for( int k : cvEntry.getValue() ) {
					samplesVal.add(samples.get(k));
					desiredVal.add(desired.get(k));
				}			
				curCost += getRMSEofLM(leafLayer, samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
			}
		}
				
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
					
					double cost = 0;
					
					if( val.v == SubmarketsByClusterLM.measure.AIC )
						cost = getAICofLM(leafLayer, samples, desired, fa);
					else {
						for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList ) {
							List<double[]> samplesTrain = new ArrayList<double[]>();
							List<double[]> desiredTrain = new ArrayList<double[]>();
							for( int k : cvEntry.getKey() ) {
								samplesTrain.add(samples.get(k));
								desiredTrain.add(desired.get(k));
							}
							
							List<double[]> samplesVal = new ArrayList<double[]>();
							List<double[]> desiredVal = new ArrayList<double[]>();
							for( int k : cvEntry.getValue() ) {
								samplesVal.add(samples.get(k));
								desiredVal.add(desired.get(k));
							}			
							cost += getRMSEofLM(leafLayer, samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
						}
					}
					double increase = cost - curCost;
																
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
			
	public static double getRMSEofLM( List<Set<double[]>> cluster, List<double[]> samplesTrain, List<double[]> desiredTrain , List<double[]> samplesVal, List<double[]> desiredVal, int[] fa) {
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
						
		} catch( Exception e ) {
			List<Integer> l = new ArrayList<Integer>();
			for( Set<double[]> s : cluster )
				l.add(s.size());
			Collections.sort(l);
			System.out.println(l);
			e.printStackTrace();
			System.exit(1);
		}
				
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
	
	public static double getAICofLM( List<Set<double[]>> cluster, List<double[]> samples, List<double[]> desired, int[] fa) {
		// check that all samples are assigned to a cluster
		Set<double[]> all = new HashSet<double[]>();
		int sumSize = 0;
		for( Set<double[]> s : cluster ) {
			sumSize+=s.size();
			all.addAll(s);
		}
		if( sumSize != all.size() )
			throw new RuntimeException("Cluster overlap!");
		
		if( !all.containsAll(samples) )
			throw new RuntimeException("Some samples not assigend to cluster!");
				
		double[] y = new double[desired.size()];
		for (int i = 0; i < desired.size(); i++)
			y[i] = desired.get(i)[0];

		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			x[i] = getStripped(d, fa);
										
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
		OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
		ols.setNoIntercept(false);
		ols.newSampleData(y, x);
		double[] beta = null;
		try {
			beta = ols.estimateRegressionParameters();
		} catch( Exception e ) {
			List<Integer> l = new ArrayList<Integer>();
			for( Set<double[]> s : cluster )
				l.add(s.size());
			Collections.sort(l);
			System.out.println(l);
			e.printStackTrace();
			System.exit(1);
		}
				
		return samples.size() * Math.log(ols.calculateResidualSumOfSquares()/samples.size()) + 2 * (beta.length + 1);
	}
	
	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
	
	public static List<Entry<List<Integer>,List<Integer>>> getCVList( int numFolds, int numRepeats, int numSamples) {
		List<Entry<List<Integer>,List<Integer>>> cvList = new ArrayList<Entry<List<Integer>,List<Integer>>>();
		for( int repeat = 0; repeat < numRepeats; repeat++ ) {			
			if( numFolds == 0 ) { // full
				List<Integer> l = new ArrayList<Integer>();
				for( int i = 0; i < numSamples; i++ )
					l.add(i);
				Collections.shuffle(l);
				List<Integer> train = new ArrayList<Integer>(l);
				List<Integer> val = new ArrayList<Integer>(l);
				cvList.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train,val) );
			} else { // n-fold cv
				List<Integer> l = new ArrayList<Integer>();
				for( int i = 0; i < numSamples; i++ )
					l.add(i);
				Collections.shuffle(l);
				int foldSize = numSamples/numFolds;
				for (int fold = 0; fold < numFolds; fold++) {				
					List<Integer> val = new ArrayList<Integer>(l.subList(fold*foldSize, (fold+1)*foldSize));
					List<Integer> train = new ArrayList<Integer>(l);
					train.removeAll(val);
					cvList.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train,val) );
				}
			}
		}
		return cvList;
	}
}
