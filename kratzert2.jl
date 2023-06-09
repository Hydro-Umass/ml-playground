using Lux, MLUtils, Optimisers, Zygote, NNlib
using Random, Statistics
using CSV, DataFrames, Printf, Dates

# read CAMELS data
gauges = CSV.File("data/camels_name.txt", delim=";") |> DataFrame
select_basin(id::Int) = any([id == b for b in [1, 3, 11, 17]])
sgauges = filter(:huc_02 => select_basin, gauges)

@enum Forcing daymet=1 maurer=2 nldas=3

function read_camels_met(id::Int, hru::Int, ds::Forcing)
    if ds == daymet
        metpath = "data/basin_dataset_public_v1p2/basin_mean_forcing/daymet"
        metfile = @sprintf("%s/%02d/%08d_lump_cida_forcing_leap.txt", metpath, hru, id)
    elseif ds == maurer
        metpath = "data/basin_dataset_public_v1p2/basin_mean_forcing/maurer"
        metfile = @sprintf("%s/%02d/%08d_lump_maurer_forcing_leap.txt", metpath, hru, id)
    else
        metpath = "data/basin_dataset_public_v1p2/basin_mean_forcing/nldas"
        metfile = @sprintf("%s/%02d/%08d_lump_nldas_forcing_leap.txt", metpath, hru, id)
    end
    header = ["Year", "Mnth", "Day", "Hr", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
    m = CSV.File(metfile, delim="\t", skipto=5, header=header, normalizenames=true) |> DataFrame
    m[!, :Date] = Date.(m[!, :Year], m[!, :Mnth], m[!, :Day])
    m
end

function read_camels_q(id::Int, hru::Int)
    qpath = "data/basin_dataset_public_v1p2/usgs_streamflow"
    qfile = @sprintf("%s/%02d/%08d_streamflow_qc.txt", qpath, hru, id)
    q = CSV.File(qfile, delim=" ", header=0, missingstring="-999.00") |> DataFrame
    q[!, :Date] = Date.(q[!, 2], q[!, 3], q[!, 4])
    q
end

function read_camels(id::Int, hru::Int, ds::Forcing=daymet)
    m = read_camels_met(id, hru, ds)
    q = read_camels_q(id, hru)
    out = innerjoin(m[!, [6, 7, 9, 10, 11, 12]], q[!, [5, 7]], on= :Date)
    out
end

data = read_camels.(sgauges[!, :gauge_id], sgauges[!, :huc_02])

# experiment 1
# x = read_camels(13340600, 17)
# dropmissing!(x)

struct RainfallRunoff{L, C} <: Lux.AbstractExplicitContainerLayer{(:lstm_cell, :estimator)}
    lstm_cell::L
    estimator::C
end

function RainfallRunoff(in_dims, hidden_dims, out_dims)
    return RainfallRunoff(LSTMCell(in_dims => hidden_dims),
                          Dense(hidden_dims => out_dims))
end

function (s::RainfallRunoff)(x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm  = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    y, st_estimator = s.estimator(y, ps.estimator, st.estimator)
    st = merge(st, (estimator=st_estimator, lstm_cell=st_lstm))
    return vec(y), st
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return mean(abs2, y_pred .- y), y_pred, st
end
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.1f0)
    return Optimisers.setup(opt, ps)
end


function train_model(x)
    train_dates(t) = (t >= Date(1983, 10, 2)) & (t < Date(2000, 10, 1))
    valid_dates(t) = (t >= Date(1999, 10, 2)) & (t < Date(2002, 10, 1))

    xt = filter(:Date => train_dates, x)
    xv = filter(:Date => valid_dates, x)

    xm = describe(xt, :mean)[:, 2]
    xs = describe(xt, :std)[:, 2]

    xtn = (xt[:, [1, 2, 3, 4, 5, 7]] .- xm[[1, 2, 3, 4, 5, 7]]') ./ xs[[1, 2, 3, 4, 5, 7]]'
    xvn = (xv[:, [1, 2, 3, 4, 5, 7]] .- xm[[1, 2, 3, 4, 5, 7]]') ./ xs[[1, 2, 3, 4, 5, 7]]'

    seq_len = 365
    x_train = zeros(Float32, 5, seq_len, size(xtn, 1)-seq_len)
    x_valid = zeros(Float32, 5, seq_len, size(xvn, 1)-seq_len)
    for t=1:size(x_train, 3)
        x_train[:, :, t] .= permutedims(xtn[t+1:t+seq_len, 1:5])
    end
    for t=1:size(x_valid, 3)
        x_valid[:, :, t] .= permutedims(xvn[t+1:t+seq_len, 1:5])
    end
    y_train = Float32.(xtn[seq_len+1:end, 6])
    y_valid = Float32.(xvn[seq_len+1:end, 6])

    train_loader = DataLoader(collect.((x_train, y_train)); batchsize=512, shuffle=true)
    valid_loader = DataLoader(collect.((x_valid, y_valid)), batchsize=512, shuffle=false)

    model = RainfallRunoff(5, 20, 1)
    # rng = Random.default_rng()
    # Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)
    opt_state = create_optimiser(ps)
    losses = Vector{Float32}()
    for epoch in 1:50
        # global ps, st, opt_state
        for (x, y) in train_loader
            (loss, y_pred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
            gs = back((one(loss), nothing, nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
            push!(losses, loss)
            println("Epoch [$epoch]: Loss $loss")
        end
    end
    return model, ps, st, xm, xs, x_valid, losses
end
