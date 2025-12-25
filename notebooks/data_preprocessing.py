import copy
from typing import Optional

import numpy as np
import pandas as pd

from sigllm.common import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.data_prep")


def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

log_step("[1/10] Load raw tables", "ratings + movies + users")
rating = pd.read_csv(
    "/data/raw/ml-1m/ratings.dat",
    header=None,
    sep="::",
    names=["uid", "iid", "rating", "timestamp"],
    engine="python",
)
item_info = pd.read_csv(
    "/data/raw/ml-1m/movies.dat",
    sep="::",
    encoding="latin-1",
    engine="python",
    names=["iid", "title", "genres"],
)
user_info = pd.read_csv(
    "/data/raw/ml-1m/users.dat",
    sep="::",
    encoding="latin-1",
    engine="python",
    names=["uid", "gender", "age", "occupation", "zipcode"],
)
log_step(
    "Raw shapes",
    f"ratings={rating.shape}, movies={item_info.shape}, users={user_info.shape}",
)

item_info.head()
user_info.head()

log_step("[2/10] Merge metadata", "content-aware enrichment (title/genres)")
rating = pd.merge(rating,item_info,on='iid',how='inner')
log_step("Merged table", f"shape={rating.shape}")
rating.head()

log_step("[3/10] Inspect time span", "temporal coverage")
date_min = pd.to_datetime(rating.timestamp,unit='s').min() 
date_max = pd.to_datetime(rating.timestamp,unit='s').max()
log_step("Time extrema", f"{date_min.date()} -> {date_max.date()}")

months_span = (date_max.year - date_min.year) * 12 + (date_max.month - date_min.month)
log_step("Total months", str(months_span))

log_step("[4/10] Build time slots", "prevent leakage via temporal split")
rating['time'] = pd.to_datetime(rating.timestamp,unit='s').map(lambda x: (x.year-date_min.year)*12+x.month)
rating.head()

rating['time'] = rating['time'] - rating['time'].min()

time_bins = np.sort(rating.time.unique())
log_step("Time bins", f"count={time_bins.shape[0]}, range=({time_bins.min()}, {time_bins.max()})")

log_step("Plot time histogram", "volume vs slot for drift check")
rating.groupby('time').agg({'rating':'count'}).reset_index().plot(x='time',kind='bar')
rating.head()

train_slot = list(range(14,24))
valid_slot = list(range(24,29))
test_slot = list(range(29,34))

log_step("[5/10] Define temporal windows", f"train={train_slot}, valid={valid_slot}, test={test_slot}")

log_step("[6/10] Create binary label", "rating >= 4 ⇒ positive=1")
rating['label'] = rating['rating'].apply(lambda x: 1 if x>=4 else 0)
rating.head()

label_stats = rating.label.describe()
log_step(
    "Label stats",
    f"mean={label_stats['mean']:.3f}, std={label_stats['std']:.3f}, positives={label_stats['mean']*100:.1f}%",
)

uid_bounds = (rating.uid.min(),rating.uid.max())
iid_bounds = (rating.iid.min(),rating.iid.max())
log_step("ID ranges", f"uid={uid_bounds}, iid={iid_bounds}")

rating_train = rating[rating['time'].isin(train_slot)].copy()
rating_valid = rating[rating['time'].isin(valid_slot)].copy()
rating_test = rating[rating['time'].isin(test_slot)].copy()

log_step(
    "Split sizes",
    f"train={rating_train.shape[0]:,}, valid={rating_valid.shape[0]:,}, test={rating_test.shape[0]:,}",
)

rating_train.shape, rating_valid.shape, rating_test.shape
rating_valid_f = rating_valid
rating_test_f = rating_test

def filter_cold_start(train,valid,test):
    log_step("Cold-start guard", "filter warm users/items only")
    train_user = train.uid.unique()
    train_item = train.iid.unique()
    valid = valid[valid['uid'].isin(train_user)]
    test = test[test['uid'].isin(train_user)]
    valid = valid[valid['iid'].isin(train_item)]
    test = test[test['iid'].isin(train_item)]
    return valid, test

log_step(
    "Validation/Test label rate",
    f"valid={rating_valid_f.label.mean():.3f}, test={rating_test_f.label.mean():.3f}"
)
log_step("Validation columns", ", ".join(rating_valid_f.columns))

def deal_with_each_u(x,u):
    items = np.array(x.iid)
    labels = np.array(x.label)
    titles = np.array(x.title)
    timestamp = np.array(x.timestamp)
    flags =  np.array(x.flag) 
    his = [0] # adding a '0' by default
    his_title = ['']
    results = []
    for i in range(items.shape[0]):
        results.append((u, items[i], timestamp[i], np.array(his), copy.copy(his_title),titles[i], labels[i], flags[i]))
        if labels[i] > 0:
            his.append(items[i])
            his_title.append(titles[i])
    return results

rating_train = rating_train.copy()

rating_train['flag'] =  pd.DataFrame(np.ones(rating_train.shape[0])*-1, index=rating_train.index)
rating_valid_f['flag'] = pd.DataFrame(np.zeros(rating_valid_f.shape[0]), index=rating_valid_f.index)
rating_test_f['flag'] = pd.DataFrame(np.ones(rating_test_f.shape[0]), index=rating_test_f.index)
log_step("[7/10] Annotate split flag", "train=-1, valid=0, test=1")
data = pd.concat([rating_train, rating_valid_f, rating_test_f],axis=0,ignore_index=True)
data = data.sort_values(by=['uid','timestamp'])
log_step("[8/10] Concatenate & sort", "order by (uid, timestamp)")
u_inter_all = data.groupby('uid').agg({'iid':list, 'label':list, 'title':list, 'timestamp':list,'flag':list})
log_step("Grouped interactions", f"users={u_inter_all.shape[0]:,}")

flag_values = data.flag.unique()
log_step("Flag uniqueness", str(flag_values))

log_step("[9/10] Build sequential samples", "deal_with_each_u")
results = []
for u in u_inter_all.index:
    results.extend(deal_with_each_u(u_inter_all.loc[u],u))

u_, i_, time_, label_, his_, his_title, title_,flag_ = [],[],[],[],[],[],[],[]
for re_ in results:
    u_.append(re_[0])
    i_.append(re_[1])
    time_.append(re_[2])
    his_.append(re_[3])
    his_title.append(re_[4])
    title_.append(re_[5])
    label_.append(re_[6])
    flag_.append(re_[7]) 

data = pd.DataFrame({"uid":u_,'iid':i_,'label':label_, 'timestamp': time_ , 'his':his_,'his_title':his_title,'title':title_, 'flag': flag_})

log_step("Sequential dataset", f"rows={data.shape[0]:,}")

data.head(2)

users = data.uid.unique()
items = data.iid.unique()
users_map = dict(zip(users, np.arange(users.shape[0])+1))
items_map = dict(zip(items, np.arange(items.shape[0])+1))

users_map[0]=0
items_map[0]=0

data['uid'] = data['uid'].map(users_map)
data['iid'] = data['iid'].map(items_map)
log_step("[10/10] Remap ids", f"users={len(users_map)-1:,}, items={len(items_map)-1:,}")
data.head(2)

data['his'] = data['his'].apply(lambda x: [items_map[k] for k in x])
hist_lens = data['his'].apply(len)
log_step(
    "History stats",
    f"avg={hist_lens.mean():.2f}, max={hist_lens.max()}, min={hist_lens.min()}"
)

final_stats = data.label.describe()
log_step(
    "Final label stats",
    f"mean={final_stats['mean']:.3f}, count={int(final_stats['count']):,}"
)

log_step("Next steps", "negative sampling + OOD tagging + serialize")

data.tail(5)