{arbor_js}


{synapse_clustering_js}


const CATMAID = {{}};


{arbor_parser_js}


const ArborParser = CATMAID.ArborParser;


class Vector3 {{
    constructor(x, y, z) {{
        this.x = x;
        this.y = y;
        this.z = z;
    }}

    distanceTo(other) {{
        let out = 0;
        for (const key of "xyz") {{
            out += Math.pow(this[key] - other[key], 2);
        }}
        return Math.sqrt(out);
    }}
}}

// ------- SHIMS FOR EXTERNAL LIBRARIES --------

const THREE = {{}};
THREE.Vector3 = Vector3;

function extend(first, other) {{
    for (const key of Object.keys(other)) {{
        first[key] = other[key];
    }}
    return first;
}}

const $ = {{}};
$.extend = extend;


// -------------- TEST CODE BEGINS HERE --------------

const fs = require('fs');

const LAMBDA = {LAMBDA};
const FRACTION = {FRACTION};
const REF_PATH = '{ref_path}';

const ARBOR_REF_PATH = '{arbor_ref_path}';
const PARSER_REF_PATH = '{parser_ref_path}';
const CLUSTERING_REF_PATH = '{clustering_ref_path}';


function writeObject(path, obj) {{
//    fs.writeFile(path, JSON.stringify(obj, null, 2), (err) => {{
//        if (err) {{
//            console.error(err);
//        }} else {{
//            console.log(`Output written to ${{path}}`)
//        }}
//    }});
    fs.writeFileSync(path, JSON.stringify(obj, null, 2));
    console.log(`Output written to ${{path}}`)
}}


function execAndWrite(root, fname, obj) {{
    writeObject(`${{root}}/${{fname}}.json`, obj);
    return obj;
}}


const arborData = {{
  url: "compact-arbor",
  json: JSON.parse(fs.readFileSync('{arbor_path}'))
}};
const skeletonData = {{
  url: "compact-skeleton",
  json: JSON.parse(fs.readFileSync('{skeleton_path}'))
}};

const realArborParser = new ArborParser();
realArborParser.init(arborData.url, arborData.json);
writeObject(PARSER_REF_PATH + '/arborparser.json', realArborParser);

const realArbor = realArborParser.arbor;
writeObject(ARBOR_REF_PATH + '/arbor.json',realArbor);

const realSynapseClustering = new SynapseClustering(
  realArbor, realArborParser.positions, realArborParser.createSynapseMap(), LAMBDA
);
writeObject(CLUSTERING_REF_PATH + '/synapseclustering.json', realSynapseClustering);

const synapseMap = realArborParser.createSynapseMap();
writeObject(PARSER_REF_PATH + '/create_synapse_map.json', synapseMap);

const locations = realArborParser.positions;
const distanceFn = (function(child, paren) {{
    return this[child].distanceTo(this[paren]);
}}).bind(locations);
const nodesDistanceTo = realArbor.nodesDistanceTo(realArbor.root, distanceFn);
writeObject(ARBOR_REF_PATH + '/nodes_distance_to.json', nodesDistanceTo);

const allSuccessors = realArbor.allSuccessors();
writeObject(ARBOR_REF_PATH + '/all_successors.json', allSuccessors);

const childrenList = realArbor.childrenArray();
writeObject(ARBOR_REF_PATH + '/children_list.json', childrenList);

const branchEndNodes = realArbor.findBranchAndEndNodes();
writeObject(ARBOR_REF_PATH + '/find_branch_and_end_nodes.json', branchEndNodes);

const flowCentrality = realArbor.flowCentrality(
    realArborParser.outputs, realArborParser.inputs,
    realArborParser.n_outputs, realArborParser.n_inputs
);
writeObject(ARBOR_REF_PATH + '/flow_centrality.json', flowCentrality);

const nodesOrderFrom = realArbor.nodesOrderFrom(realArbor.root);
writeObject(ARBOR_REF_PATH + '/nodes_order_from.json', nodesOrderFrom);

const partitions = realArbor.partition();
writeObject(ARBOR_REF_PATH + '/partition.json', partitions);

const distanceMap = realSynapseClustering.distanceMap();
writeObject(CLUSTERING_REF_PATH + '/distance_map.json', distanceMap);

const densityHillMap = realSynapseClustering.densityHillMap();
writeObject(CLUSTERING_REF_PATH + '/density_hill_map.json', densityHillMap);

const clusters = realSynapseClustering.clusters(densityHillMap);
writeObject(CLUSTERING_REF_PATH + '/clusters.json', distanceMap);

const clusterMaps = realSynapseClustering.clusterMaps(densityHillMap);
writeObject(CLUSTERING_REF_PATH + '/cluster_maps.json', distanceMap);

const clusterSizes = execAndWrite(
    CLUSTERING_REF_PATH, 'cluster_sizes', realSynapseClustering.clusterSizes(densityHillMap)
);

const segregationIndex = execAndWrite(
    CLUSTERING_REF_PATH, 'segregation_index',
    realSynapseClustering.segregationIndex(clusters, realArborParser.outputs, realArborParser.inputs)
);

const arborRegions = execAndWrite(
    CLUSTERING_REF_PATH, 'find_arbor_regions',
    realSynapseClustering.findArborRegions(realArbor, flowCentrality, FRACTION)
);

const findAxon = execAndWrite(
    CLUSTERING_REF_PATH, 'find_axon',
    realSynapseClustering.findAxon(realArborParser, FRACTION, realArborParser.positions)
);

const findAxonCut = execAndWrite(
    CLUSTERING_REF_PATH, 'find_axon_cut',
    realSynapseClustering.findAxonCut(realArbor, realArborParser.outputs, arborRegions.above, realArborParser.positions)
);
