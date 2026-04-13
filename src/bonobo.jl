# =============================================================================
# Part of this file is copied/adapted from Bonobo.jl (MIT License)
# Original repository: https://wikunia.github.io/Bonobo.jl
# License: MIT License
# =============================================================================

"""
    AbstractNode

The abstract type for a tree node. Your own type for `Node` given to [`initialize`](@ref) needs to subtype it.
The default if you don't provide your own is [`DefaultNode`](@ref).
"""
abstract type AbstractNode end

"""
    AbstractSolution{Node<:AbstractNode, Value}

The abstract type for a `Solution` object. The default is [`DefaultSolution`](@ref).
It is parameterized by `Node` and `Value` where `Value` is the value which describes the full solution i.e the value for every variable.
"""
abstract type AbstractSolution{Node<:AbstractNode,Value} end

"""
    BnBNodeInfo

Holds the necessary information of every node.
This needs to be added by every `AbstractNode` as `std::BnBNodeInfo`

```julia
id :: Int
lb :: Float64
ub :: Float64
depth :: Int
```
"""
mutable struct BnBNodeInfo
    id::Int
    lb::Float64
    ub::Float64
    depth::Int
end

"""
    DefaultNode <: AbstractNode

The default structure for saving node information.
Currently this includes only the necessary `std::BnBNodeInfo` which needs to be part of every [`AbstractNode`](@ref).
"""
mutable struct DefaultNode <: AbstractNode
    std::BnBNodeInfo
end

"""
    DefaultSolution{Node<:AbstractNode,Value} <: AbstractSolution{Node, Value}

The default struct to save a solution of the branch and bound run.
It holds
```julia
objective :: Float64
solution  :: Value
node      :: Node
```
Both the `Value` and the `Node` type are determined by the [`initialize`](@ref) method.

`solution` holds the information to obtain the solution for example the values of all variables.
"""
mutable struct DefaultSolution{Node<:AbstractNode,Value} <: AbstractSolution{Node,Value}
    objective::Float64
    solution::Value
    node::Node
end

"""
    AbstractTraverseStrategy

The abstract type for a traverse strategy. 
If you implement a new traverse strategy this must be the supertype. 

If you want to implement your own strategy the [`get_next_node`](@ref) function needs a new method 
which dispatches on the `traverse_strategy` argument. 
"""
abstract type AbstractTraverseStrategy end


"""
    AbstractBranchStrategy

The abstract type for a branching strategy. 
If you implement a new branching strategy, this must be the supertype. 

If you want to implement your own strategy, you must implement a new method for [`get_branching_variable`](@ref)
which dispatches on the `branch_strategy` argument. 
"""
abstract type AbstractBranchStrategy end

"""
    BestFirstSearch <: AbstractTraverseStrategy

The BestFirstSearch traverse strategy always picks the node with the lowest bound first.
If there is a tie then the smallest node id is used as a tie breaker.
"""
struct BestFirstSearch <: AbstractTraverseStrategy end

struct DepthFirstSearch <: AbstractTraverseStrategy end

@deprecate BFS() BestFirstSearch() false

"""
    FIRST <: AbstractBranchStrategy

The `FIRST` strategy always picks the first variable which isn't fixed yet and can be branched on.
"""
struct FIRST <: AbstractBranchStrategy end

"""
    MOST_INFEASIBLE <: AbstractBranchStrategy

The `MOST_INFEASIBLE` strategy always picks the variable which is furthest away from being "fixed" and can be branched on.
"""
struct MOST_INFEASIBLE <: AbstractBranchStrategy end

mutable struct Options
    traverse_strategy::AbstractTraverseStrategy
    branch_strategy::AbstractBranchStrategy
    atol::Float64
    rtol::Float64
    dual_gap_limit::Float64
    abs_gap_limit::Float64
end

"""
    BnBTree{Node<:AbstractNode,Root,Value,Solution<:AbstractSolution{Node,Value}}

Holds all the information of the branch and bound tree. 

```
incumbent::Float64 - The best objective value found so far. Is stores as problem is a minimization problem
incumbent_solution::Solution - The currently best solution object
lb::Float64        - The highest current lower bound 
solutions::Vector{Solution} - A list of solutions
node_queue::PriorityQueue{Int,Tuple{Float64, Int}} - A priority queue with key being the node id and the priority consists of the node lower bound and the node id.
nodes::Dict{Int, Node}  - A dictionary of all nodes with key being the node id and value the actual node.
root::Root      - The root node see [`set_root!`](@ref)
branching_indices::Vector{Int} - The indices to be able to branch on used for [`get_branching_variable`](@ref)
num_nodes::Int  - The number of nodes created in total
sense::Symbol   - The objective sense: `:Max` or `:Min`.
options::Options  - All options for the branch and bound tree. See [`Options`](@ref).
```
"""
mutable struct BnBTree{Node<:AbstractNode,Root,Value,Solution<:AbstractSolution{Node,Value}}
    incumbent::Float64
    incumbent_solution::Union{Nothing,Solution}
    lb::Float64
    solutions::Vector{Solution}
    node_queue::PriorityQueue{Int,Tuple{Float64,Int}}
    nodes::Dict{Int,Node}
    root::Root
    branching_indices::Vector{Int}
    num_nodes::Int
    sense::Symbol
    options::Options
end

Base.broadcastable(x::BnBTree) = Ref(x)

"""
    initialize(; kwargs...)

Initialize the branch and bound framework with the the following arguments.
Later it can be dispatched on `BnBTree{Node, Root, Solution}` for various methods.

# Keyword arguments
- `traverse_strategy` [`BestFirstSearch`] currently the only supported traverse strategy is [`BestFirstSearch`](@ref). Should be an [`AbstractTraverseStrategy`](@ref)
- `branch_strategy` [`FIRST`] currently the only supported branching strategies are [`FIRST`](@ref) and [`MOST_INFEASIBLE`](@ref). Should be an [`AbstractBranchStrategy`](@ref)
- `atol` [1e-6] the absolute tolerance to check whether a value is discrete
- `rtol` [1e-6] the relative tolerance to check whether a value is discrete
- `Node` [`DefaultNode`](@ref) can be special structure which is used to store all information about a node. 
    - needs to have `AbstractNode` as the super type
    - needs to have `std :: BnBNodeInfo` as a field (see [`BnBNodeInfo`](@ref))
- `Solution` [`DefaultSolution`](@ref) stores the node and several other information about a solution
- `root` [`nothing`] the information about the root problem. The type can be used for dispatching on types 
- `sense` [`:Min`] can be `:Min` or `:Max` depending on the objective sense
- `Value` [`Vector{Float64}`] the type of a solution  

Return a [`BnBTree`](@ref) object which is the input for [`optimize!`](@ref).
"""
function initialize(;
    traverse_strategy=BestFirstSearch(),
    branch_strategy=FIRST(),
    atol=1e-6,
    rtol=1e-6,
    Node=DefaultNode,
    Value=Vector{Float64},
    Solution=DefaultSolution{Node,Value},
    root=nothing,
    sense=:Min,
    dual_gap_limit=1e-5,
    abs_gap_limit=1e-5,
)
    return BnBTree{Node,typeof(root),Value,Solution}(
        Inf,
        nothing,
        -Inf,
        Vector{Solution}(),
        PriorityQueue{Int,Tuple{Float64,Int}}(),
        Dict{Int,Node}(),
        root,
        get_branching_indices(root),
        0,
        sense,
        Options(traverse_strategy, branch_strategy, atol, rtol, dual_gap_limit, abs_gap_limit),
    )
end

"""
    get_branching_indices(root)

Return a vector of variables to branch on from the current root object.
"""
function get_branching_indices end

"""
    sort_solutions!(solutions::Vector{<:AbstractSolution})

Sort the solutions vector by objective value such that the best solution is at index 1.
"""
function sort_solutions!(solutions::Vector{<:AbstractSolution})
    return sort!(solutions; by=s -> s.objective)
end

"""
    set_node_bound!(objective_sense::Symbol, node::AbstractNode, lb, ub)

Set the bounds of the `node` object to the lower and upper bound given. 
Internally everything is stored as a minimization problem. Therefore the objective_sense `:Min`/`:Max` is needed.
"""
function set_node_bound!(objective_sense::Symbol, node::AbstractNode, lb, ub)
    if isnan(ub)
        ub = Inf
    end
    if objective_sense == :Min
        node.lb = max(lb, node.lb)
        node.ub = ub
    else
        node.lb = max(-lb, node.lb)
        node.ub = -ub
    end
end

"""
    bound!(tree::BnBTree, current_node_id::Int)

Close all nodes which have a lower bound higher or equal to the incumbent
"""
function bound!(tree::BnBTree, current_node_id::Int)
    for (_, node) in tree.nodes
        if node.id != current_node_id && node.lb >= tree.incumbent
            close_node!(tree, node)
        end
    end
end

"""
    close_node!(tree::BnBTree, node::AbstractNode)

Delete the node from the nodes dictionary and the priority queue.
"""
function close_node!(tree::BnBTree, node::AbstractNode)
    delete!(tree.nodes, node.id)
    return delete!(tree.node_queue, node.id)
end

"""
    branch!(tree, node)

Get the branching variable with [`get_branching_variable`](@ref) and then calls [`get_branching_nodes_info`](@ref) and [`add_node!`](@ref).
"""
function branch!(tree, node)
    variable_idx = get_branching_variable(tree, tree.options.branch_strategy, node)
    # no branching variable selected => return
    variable_idx == -1 && return
    nodes_info = get_branching_nodes_info(tree, node, variable_idx)
    for node_info in nodes_info
        add_node!(tree, node, node_info)
    end
end

"""
    get_branching_variable(tree::BnBTree, ::MOST_INFEASIBLE, node::AbstractNode)

Return the branching variable which is furthest away from being feasible based on [`get_distance_to_feasible`](@ref)
or `-1` if all integer constraints are respected.
"""
function get_branching_variable(tree::BnBTree, ::MOST_INFEASIBLE, node::AbstractNode)
    values = get_relaxed_values(tree, node)
    best_idx = -1
    max_distance_to_feasible = 0.0
    for i in tree.branching_indices
        value = values[i]
        if !is_approx_feasible(tree, value)
            distance_to_feasible = get_distance_to_feasible(tree, value)
            if distance_to_feasible > max_distance_to_feasible
                best_idx = i
                max_distance_to_feasible = distance_to_feasible
            end
        end
    end
    return best_idx
end

"""
    get_branching_nodes_info(tree::BnBTree, node::AbstractNode, vidx::Int)

Create the information for new branching nodes based on the variable index `vidx`.
Return a list of those information as a `NamedTuple` vector.

# Example
The following would add the necessary information about a new node and return it. The necessary information are the fields required by the [`AbstractNode`](@ref).
For this examle the required fields are the lower and upper bounds of the variables as well as the status of the node.
```julia
nodes_info = NamedTuple[]
push!(nodes_info, (
    lbs = lbs,
    ubs = ubs,
    status = MOI.OPTIMIZE_NOT_CALLED,
))
return nodes_info
```
"""
function get_branching_nodes_info end

"""
    set_root!(tree::BnBTree, node_info::NamedTuple)

Set the root node information based on the `node_info` which needs to include the same fields as the `Node` struct given 
to the [`initialize`](@ref) method. (Besides the `std` field which is set by Bonobo automatically)

# Example
If your node structure is the following:
```julia
mutable struct MIPNode <: AbstractNode
    std :: BnBNodeInfo
    lbs :: Vector{Float64}
    ubs :: Vector{Float64}
    status :: MOI.TerminationStatusCode
end
```

then you can call the function with this syntax:

```julia
set_root!(tree, (
    lbs = fill(-Inf, length(x)),
    ubs = fill(Inf, length(x)),
    status = MOI.OPTIMIZE_NOT_CALLED
))
```
"""
function set_root!(tree::BnBTree, node_info::NamedTuple)
    return add_node!(tree, nothing, node_info)
end

"""
    add_node!(tree::BnBTree{Node}, parent::Union{AbstractNode, Nothing}, node_info::NamedTuple)

Add a new node to the tree using the `node_info`. For information on that see [`set_root!`](@ref).
"""
function add_node!(
    tree::BnBTree{Node},
    parent::Union{AbstractNode,Nothing},
    node_info::NamedTuple,
) where {Node<:AbstractNode}
    node_id = tree.num_nodes + 1
    node = create_node(Node, node_id, parent, node_info)
    # only add the node if it's better than the current best solution
    if node.lb < tree.incumbent
        tree.nodes[node_id] = node
        tree.node_queue[node_id] = (node.lb, node_id)
        tree.num_nodes += 1
    end
end

"""
    create_node(Node, node_id::Int, parent::Union{AbstractNode, Nothing}, node_info::NamedTuple)

Creates a node of type `Node` with id `node_id` and the named tuple `node_info`. 
For information on that see [`set_root!`](@ref).
"""
function create_node(Node, node_id::Int, parent::Union{AbstractNode,Nothing}, node_info::NamedTuple)
    lb = -Inf
    depth = 1
    if !isnothing(parent)
        lb = parent.lb
        depth = parent.depth + 1
    end
    bnb_node = structfromnt(BnBNodeInfo, (id=node_id, lb=lb, ub=Inf, depth=depth))
    bnb_nt = (std=bnb_node,)
    node_nt = merge(bnb_nt, node_info)
    return structfromnt(Node, node_nt)
end

"""
    get_next_node(tree::BnBTree, ::BestFirstSearch)

Get the next node of the tree which shall be evaluted next by [`evaluate_node!`](@ref).
If you want to implement your own traversing strategy check out [`AbstractTraverseStrategy`](@ref).
"""
function get_next_node(tree::BnBTree, ::BestFirstSearch)
    node_id, _ = first(tree.node_queue)
    return tree.nodes[node_id]
end

function get_next_node(tree::BnBTree, ::DepthFirstSearch)
    node_id = argmax(k -> tree.nodes[k].depth, keys(tree.nodes))
    return tree.nodes[node_id]
end

"""
    evaluate_node!(tree, node)

Evaluate the current node and return the lower and upper bound of that node.
"""
function evaluate_node! end

#=
    Access standard AbstractNode internals without using .std syntax
=#
@inline function Base.getproperty(c::AbstractNode, s::Symbol)
    if s in (:id, :lb, :ub, :depth)
        Core.getproperty(Core.getproperty(c, :std), s)
    else
        getfield(c, s)
    end
end

@inline function Base.setproperty!(c::AbstractNode, s::Symbol, v)
    if s in (:id, :lb, :ub, :depth)
        Core.setproperty!(c.std, s, v)
    else
        Core.setproperty!(c, s, v)
    end
end

"""
    is_approx_feasible(tree::BnBTree, value)

Return whether a given `value` is approximately feasible based on the tolerances defined in the tree options. 
"""
function is_approx_feasible(tree::BnBTree, value::Number)
    return is_approx_feasible(value; atol=tree.options.atol, rtol=tree.options.rtol)
end

function is_approx_feasible(value::Number; atol=1e-6, rtol=1e-6)
    return isapprox(value, round(value); atol, rtol)
end

"""
    get_distance_to_feasible(tree::BnBTree, value)

Return the distance of feasibility for the given value.

- if `value::Number` this returns the distance to the nearest discrete value
"""
function get_distance_to_feasible(tree::BnBTree, value::Number)
    return abs(round(value) - value)
end

export BnBTree, BnBNodeInfo, AbstractNode, AbstractSolution
export AbstractTraverseStrategy, AbstractBranchStrategy
export BestFirstSearch, DepthFirstSearch
export FIRST, MOST_INFEASIBLE
